package ai.certifai.solution.convolution.objectdetection.transferlearning;

import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.transform.ColorConversionTransform;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.deeplearning4j.nn.layers.objdetect.YoloUtils;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.TinyYOLO;
import org.deeplearning4j.zoo.model.YOLO2;
import org.deeplearning4j.zoo.util.darknet.COCOLabels;
import org.deeplearning4j.zoo.util.darknet.VOCLabels;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;

import java.awt.event.KeyEvent;
import java.util.ArrayList;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_core.flip;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.bytedeco.opencv.global.opencv_imgproc.LINE_8;
import static org.nd4j.linalg.ops.transforms.Transforms.euclideanDistance;

public class WebCam {
    
    //Camera position change between "front" and "back"
    //front camera requires flipping of the image
    private static String cameraPos = "front";

    //swap between camera with 0 -? on the parameter
    //Default is 0
    private static int cameraNum = 0;
    private static Thread thread;
    private static final int gridWidth = 13;
    private static final int gridHeight = 13;
    private static double detectionThreshold = 0.5;
    private static final int tinyyolowidth = 416;
    private static final int tinyyoloheight = 416;


    private static int safeDistance = 300;



    public static void main(String[] args) throws Exception {
        if (!cameraPos.equals("front") && !cameraPos.equals("back")) {
            throw new Exception("Unknown argument for camera position. Choose between front and back");
        }

        FrameGrabber grabber = FrameGrabber.createDefault(cameraNum);
        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();

        grabber.start();
        String winName = "Object Detection";
        CanvasFrame canvas = new CanvasFrame(winName);
        int w = grabber.getImageWidth();
        int h = grabber.getImageHeight();
        canvas.setCanvasSize(w, h);
        ZooModel model = TinyYOLO.builder().numClasses(0).build();
        ComputationGraph initializedModel = (ComputationGraph) model.initPretrained();
        NativeImageLoader loader = new NativeImageLoader(tinyyolowidth, tinyyoloheight, 3, new ColorConversionTransform(COLOR_BGR2RGB));
        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
        VOCLabels labels = new VOCLabels();
//        COCOLabels labels = new COCOLabels();
        while (true) {
            Frame frame = grabber.grab();

            //if a thread is null, create new thread
            if (thread == null) {
                thread = new Thread(() ->
                {
                    while (frame != null) {
                        try {
                            Mat rawImage = new Mat();
                            //Flip the camera if opening front camera
                            if (cameraPos.equals("front")) {
                                Mat inputImage = converter.convert(frame);
                                flip(inputImage, rawImage, 1);
                            } else {
                                rawImage = converter.convert(frame);
                            }
                            Mat resizeImage = new Mat();
                            resize(rawImage, resizeImage, new Size(tinyyolowidth, tinyyoloheight));
                            INDArray inputImage = loader.asMatrix(resizeImage);
                            scaler.transform(inputImage);
                            INDArray outputs = initializedModel.outputSingle(inputImage);
                            List<DetectedObject> objs = YoloUtils.getPredictedObjects(Nd4j.create(((TinyYOLO) model).getPriorBoxes()), outputs, detectionThreshold, 0.4);

                            List<INDArray> centers = new ArrayList<>();
                            List<INDArray> people = new ArrayList<>();
                            int violators = 0;

                            int centerX;
                            int centerY;

                            int x1=0,x2=0,y1=0,y2=0;
                            int bodywidth = 40;
                            int focalpoint = 606;

                            for (DetectedObject obj : objs) {
                                if (obj.getPredictedClass() == 14) {
                                    double[] xy1 = obj.getTopLeftXY();
                                    double[] xy2 = obj.getBottomRightXY();
                                    String label = labels.getLabel(obj.getPredictedClass());
                                    x1 = (int) Math.round(w * xy1[0] / gridWidth);
                                    y1 = (int) Math.round(h * xy1[1] / gridHeight);
                                    x2 = (int) Math.round(w * xy2[0] / gridWidth);
                                    y2 = (int) Math.round(h * xy2[1] / gridHeight);

                                    centerX = (x1 + x2) / 2;
                                    centerY = (y1 + y2) / 2;

                                    rectangle(rawImage, new Point(x1, y1), new Point(x2, y2), Scalar.GREEN, 2, 0, 0);
                                    putText(rawImage, label, new Point(x1 + 2, y2 - 2), FONT_HERSHEY_DUPLEX, 1, Scalar.GREEN);
                                    circle(rawImage, new Point(centerX, centerY), 2, new Scalar(0, 255, 0, 0), -1, 0, 0);

                                    centers.add(Nd4j.create(new float[]{(float) centerX, (float) centerY}));
                                    people.add(Nd4j.create(new float[]{(float) xy1[0], (float) xy1[1], (float) xy2[0], (float) xy2[1]}));
                                }
                            }
                            for (int i = 0; i < centers.size(); i++) {
                                for (int j = 0; j < centers.size(); j++) {

                                    int xPixeli = people.get(i).getInt(2) -people.get(i).getInt(0)+1; //number of pixel in box
                                    int xPixelj = people.get(j).getInt(2) -people.get(j).getInt(0)+1;
                                    double xdistance = euclideanDistance(centers.get(i), centers.get(j));//pixel xdistance in between two person without depth
                                    int initDepthi = bodywidth*focalpoint/xPixeli; //depth for person i
                                    int initDepthj = bodywidth*focalpoint/xPixelj; //depth for person j

                                    int yDistance = initDepthi-initDepthj;  //between person i and j
                                    double xyDistance = Math.pow(Math.pow((double)yDistance,2.0) + Math.pow(xdistance,2.0) , 0.5); //eudclidean distance in 3D

                                    if (xyDistance < safeDistance && xyDistance > 0) {
                                        line(rawImage, new Point(centers.get(i).getInt(0), centers.get(i).getInt(1)),
                                                new Point(centers.get(j).getInt(0), centers.get(j).getInt(1)), Scalar.RED, 2, 1, 0);

//
                                        int xmin = people.get(i).getInt(0);
                                        int ymin = people.get(i).getInt(1);
                                        int xmax = people.get(j).getInt(2);
                                        int ymax = people.get(j).getInt(3);

                                        violators ++;

                                        rectangle(rawImage, new Point(xmin, ymin), new Point(xmax, ymax), Scalar.RED, 2, LINE_8, 0);
                                        circle(rawImage, new Point(centers.get(i).getInt(0), centers.get(i).getInt(1)), 3, Scalar.RED, -1, 0, 0);
                                    }
                                }
                            }
                            putText(rawImage, String.format("Number of people: %d", people.size()), new Point(10, 30), 4, 1.0, new Scalar(255, 0, 0, 0), 2, LINE_8, false);
                            putText(rawImage, String.format("Number of violators: %d", violators), new Point(10, 60), 4, 1.0, new Scalar(0, 0, 255, 0), 2, LINE_8, false);
                            canvas.showImage(converter.convert(rawImage));
                        } catch (Exception e) {
                            throw new RuntimeException(e);
                        }
                    }
                });
                thread.start();
            }
            KeyEvent t = canvas.waitKey(33);
            if ((t != null) && (t.getKeyCode() == KeyEvent.VK_Q)) {
                break;
            }
        }
        canvas.dispose();
    }
}

