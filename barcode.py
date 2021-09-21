import io
import os
import cv2
import sys
import time
import argparse
import picamera
import threading
import numpy as np

BARCODE_TYPE = {
    "NONE": cv2.barcode.NONE,
    "EAN_8" : cv2.barcode.EAN_8,
    "EAN_13": cv2.barcode.EAN_13,
    "UPC_A": cv2.barcode.UPC_A,
    "UPC_E": cv2.barcode.UPC_E,
    "UPC_EAN_EXTENSION": cv2.barcode.UPC_EAN_EXTENSION,
}

class Visualizer(threading.Thread):
    def __init__(self, owner):
        super(Visualizer, self).__init__()
        self.owner = owner
        self.timestamp = None
        self.current_timestamp = None
        self.image = None
        self.terminated = False
        self.start()

    def run(self):
        while not self.terminated:
            if self.timestamp:
                if self.current_timestamp:
                    if self.timestamp > self.current_timestamp:
                        cv2.imshow("Frame", self.image)
                        self.current_timestamp = self.timestamp
                else:
                    self.current_timestamp = self.timestamp
            if cv2.waitKey(33) == 27:
                self.terminated = True
        self.owner.done = True

class Barcode:
    def __init__(self, binfo, btype, points):
        self.info = binfo
        self.type = btype
        self.points = points

    def __str__(self):
        return str(self.info) + " " + str(self.type)

    def Draw(self, image):
        p1 = np.array(self.points[0], dtype=int)
        p2 = np.array(self.points[2], dtype=int)
        cv2.rectangle(image, p1, p2, (255, 0, 0))
        cv2.putText(image, "{}".format(self.info), p2, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1, cv2.LINE_AA)

class ImageProcessor(threading.Thread):
    def __init__(self, owner):
        super(ImageProcessor, self).__init__()
        self.stream = io.BytesIO()
        self.event = threading.Event()
        self.terminated = False
        self.owner = owner
        self.Reset()
        self.start()

    def run(self):
        # This method runs in a separate thread
        while not self.terminated:
            # Wait for an image to be written to the stream
            if self.event.wait(1):
                try:
                    self.stream.seek(0)
                    self.image = cv2.imdecode(np.frombuffer(self.stream.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)
                    retval, decoded_info, decoded_type, points = self.detector.detectAndDecode(self.image)
                    if retval:
                        for binfo, btype, brect in zip(decoded_info, decoded_type, points):
                            self.barcodes.append(Barcode(binfo, btype, brect))
                finally:
                    # Reset the stream and event
                    self.stream.seek(0)
                    self.stream.truncate()
                    self.event.clear()
                    # Return ourselves to the available pool
                    with self.owner.lock:
                        self.done = True
                        self.owner.pool.append(self)
    
    def Reset(self):
        self.done = False
        self.image = None
        self.timestamp = None
        self.frame = 0
        self.detector = cv2.barcode_BarcodeDetector()
        self.barcodes = []

class ProcessOutput(object):
    def __init__(self, args):
        self.done = False
        # Construct a pool of image processors along with a lock
        # to control access between threads
        self.lock = threading.Lock()
        self.pool = [ImageProcessor(self) for i in range(args.num_threads)]
        self.processor = None
        self.frame = 0

        self.args = args
        if self.args.visualize:
            self.visualizer = Visualizer(self)

    def PrintResult(self, processor):
        # assert(processor.done)
        print(processor.timestamp, processor.frame, end=" ")
        for barcode in processor.barcodes:
            print(barcode)
        print(flush=True)

    def ShowResult(self, processor):
        # assert(processor.done)
        self.visualizer.timestamp = processor.timestamp
        self.visualizer.image = processor.image

    def StoreResult(self, processor):
        # assert(processor.done)
        dirName = "barcode"

        if not os.path.isdir(dirName):
            os.mkdir(dirName)

        cv2.imwrite("{}/{}_{}.jpg".format(dirName, processor.timestamp, processor.frame), processor.image)

    def write(self, buf):
        if buf.startswith(b'\xff\xd8'):
            # New frame; set the current processor going and grab a spare one
            self.frame += 1
            timestamp = str(round(time.time() * 1000))
            if self.processor:
                self.processor.event.set()
            with self.lock:
                if self.pool:
                    if self.pool[-1].done:
                        if self.args.visualize or self.args.store:
                            for barcode in self.pool[-1].barcodes:
                                barcode.Draw(self.pool[-1].image)
                        if self.args.print_results:
                            self.PrintResult(self.pool[-1])
                        if self.args.visualize:
                            self.ShowResult(self.pool[-1])
                        if self.args.store:
                            self.StoreResult(self.pool[-1])
                        self.pool[-1].Reset()
                    self.processor = self.pool.pop()
                    self.processor.frame = self.frame
                    self.processor.timestamp = timestamp
                else:
                    # No processor's available, we'll have to skip
                    # this frame; you may want to print a warning
                    # here to see whether you hit this case
                    if self.args.print_results:
                        print(timestamp, self.frame, flush=True)
                    self.processor = None
        if self.processor:
            self.processor.stream.write(buf)

    def Flush(self):
        # When told to flush (this indicates end of recording), shut
        # down in an orderly fashion. First, add the current processor
        # back to the pool
        print("Terminating ...")
        if self.processor:
            with self.lock:
                self.pool.append(self.processor)
                self.processor = None

        if self.args.visualize:
            self.visualizer.terminated = True
            self.visualizer.join()

        # Now, empty the pool, joining each thread as we go
        while True:
            proc = None
            with self.lock:
                try:
                    proc = self.pool.pop()
                except IndexError:
                    pass # pool is empty
            if not proc:
                break
            proc.terminated = True
            proc.join()

def ParseCommandLineArguments():
    parser = argparse.ArgumentParser(description='Detects barcodes and decodes their values.')
    parser.add_argument('-n', '--num-threads', default=1, type=int)
    parser.add_argument('-p', '--print-results', action='store_const', const=True, default=False)
    parser.add_argument('-v', '--visualize', action='store_const', const=True, default=False)
    parser.add_argument('-s', '--store', action='store_const', const=True, default=False)
    parser.add_argument('-iw', '--image-width', default=640, type=int)
    parser.add_argument('-ih', '--image-height', default=480, type=int)
    parser.add_argument('-f', '--fps', default=30, type=int)
    return parser.parse_args(sys.argv[1:])

if __name__ == "__main__":
    args = ParseCommandLineArguments()

    with picamera.PiCamera(resolution=(args.image_width, args.image_height), framerate=args.fps) as camera:
        time.sleep(2)
        output = ProcessOutput(args)
        camera.start_recording(output, format='mjpeg')
        try:
            while not output.done:
                camera.wait_recording()
        except KeyboardInterrupt:
            pass
        camera.stop_recording()
        output.Flush()

