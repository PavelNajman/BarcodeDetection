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

class Barcode:
    """
    A class used to represent a barcode.

    Attributes
    ----------
    info : str
        decoded barcode value
    type : int
        a type of barcode (e.g. EAN-13)
    points : numpy.array
        vertices of barcode rectangle

    Methods
    -------
    Draw(image)
        Draws barcode's rectangle and its value to the given image.
    """

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
    """
    Thread that detects and decodes barcodes in a given image.

    Attributes
    ----------
        stream
            A byte stream that holds the image data.
        event
            A threading event that is set when the whole frame is available and which starts
            the image processing.
        terminated
            A flag that signifies the termination of the image processing thread.
        owner
            An objects that spawns and owns the image processing thread.
        barcodes
            A list of barcode objects.
        done
            A flag that signifies the end of the processing of the current image.
        image
            An image that is currently processed.
        timestamp
            A timestamp of the currently processed image.
        frame
            A frame sequence number.
        detector
            An instance of a barcode detector.

    Method
    ------
        run()
            Image processing main loop in which the thread wait for the whole frame to be read and
            then it detects and decodes the barcodes in it.
        reset()
            Resets the state of the thread.
    """

    def __init__(self, owner):
        super(ImageProcessor, self).__init__()
        self.stream = io.BytesIO()
        self.event = threading.Event()
        self.terminated = False
        self.owner = owner
        self.reset()
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
    
    def reset(self):
        self.done = False
        self.image = None
        self.timestamp = None
        self.frame = 0
        self.detector = cv2.barcode_BarcodeDetector()
        self.barcodes = []

class ProcessOutput(object):
    """
    Handles frame readout and work distribution among threads.

    Attributes
    ----------
        done
            A flag that signifies the end of the processing.
        lock
            A threading lock used for synchronization.
        pool
            A pool of available image processing workers.
        processor
            An image processing worker that currently waits for a whole frame to be read.
        frame
            A frame counter.
        args
            A parsed command line arguments.
        visualizer
            A visualization thread.

    Methods
    -------
        print_result(processor)
            Prints the results of the worker that has finished the image processing.
        show_result(processor)
            Visualizes the results of the worker that has finished the image processing.
        store_result(processor)
            Stores the results of the worker that has finished the image processing.
        write(buf)
            Reads the frame from the camera and assigns it to the available worker.
        flush()
            Shuts down in an orderly fashion.
    """

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

    def print_result(self, processor):
        # assert(processor.done)
        print(processor.timestamp, processor.frame, end=" ")
        for barcode in processor.barcodes:
            print(barcode)
        print(flush=True)

    def show_result(self, processor):
        # assert(processor.done)
        self.visualizer.timestamp = processor.timestamp
        self.visualizer.image = processor.image

    def store_result(self, processor):
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
                            self.print_result(self.pool[-1])
                        if self.args.visualize:
                            self.show_result(self.pool[-1])
                        if self.args.store:
                            self.store_result(self.pool[-1])
                        self.pool[-1].reset()
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

    def flush(self):
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

class Visualizer(threading.Thread):
    """
    Thread that shows images with drawn barcode rectangles and values.

    Attributes
    ----------
        owner
            An object that spawn and owns the visualization thread.
        timestamp
            A timestamp of an image that is to be shown.
        current_timestamp
            A timestamp of an image that is currently shown.
        image
            An image that is to be shown.
        terminated
            A flag that signifies the termination of the visualization thread.

    Methods
    -------
        run()
            Visualization main loop that shows the given image if its timestamp
            is greater than the timestamp of currently shown image. This loop is
            terminated when ESC is pressed.
    """

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

def ParseCommandLineArguments():
    """ Parses command line arguments. """

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
        output.flush()

