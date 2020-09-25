package ai.certifai.solution.convolution.objectdetection;

import org.bytedeco.javacv.*;

import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_core.Mat;

import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;
import static org.nd4j.linalg.ops.transforms.Transforms.euclideanDistance;
import static org.nd4j.linalg.util.MathUtils.sigmoid;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.deeplearning4j.nn.layers.objdetect.YoloUtils;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.YOLO2;
import org.deeplearning4j.zoo.util.darknet.COCOLabels;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.awt.event.KeyEvent;
import java.util.ArrayList;
import java.util.List;


public class Video {
    private static ComputationGraph model;
    private static int numClass = 80;
    private static int[][] anchors = {{10, 13}, {16, 30}, {33, 23}, {30, 61}, {62, 45}, {59, 119}, {116, 90}, {156, 198}, {373, 326}};
    private static final int gridWidth = 13;
    private static final int gridHeight = 13;
    private static double detectionThreshold = 0.2;
    private static final int tinyyoloWidth = 416;
    private static final int tinyyoloHeight = 416;

    public static void main(String[] args) throws Exception {

        int safeDistance = 80;

        String videoPath = ("C:\\Users\\User\\Downloads\\Test.mp4");
        FFmpegFrameGrabber grabber = new FFmpegFrameGrabber(videoPath);
        grabber.setFormat("mp4");
        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
        grabber.start();

        String winName = "Object Detection";
        CanvasFrame canvas = new CanvasFrame(winName);

        ZooModel model = YOLO2.builder().numClasses(0).build();
        ComputationGraph initializedModel = (ComputationGraph) model.initPretrained();

        NativeImageLoader nil = new NativeImageLoader(tinyyoloHeight, tinyyoloWidth, 3);
        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
        COCOLabels labels = new COCOLabels();

        System.out.println("Start running video");

        while ((grabber.grab()) != null) {
            Frame frame = grabber.grabImage();

            Mat opencvMat = converter.convert(frame);
            int w = opencvMat.cols();
            int h = opencvMat.rows();

            INDArray inputImage = nil.asMatrix(frame);
            scaler.transform(inputImage);

            INDArray outputs = initializedModel.outputSingle(inputImage);
            List<DetectedObject> objs = YoloUtils.getPredictedObjects(Nd4j.create(((YOLO2) model).getPriorBoxes()), outputs, detectionThreshold, 0.4);
            List<INDArray> centers = new ArrayList<>();
            List<INDArray> people = new ArrayList<>();
            List<INDArray> violators = new ArrayList<>();

            int centerX;
            int centerY;

            for (DetectedObject obj : objs) {
                if (obj.getPredictedClass() == 0) {
                    double[] xy1 = obj.getTopLeftXY();
                    double[] xy2 = obj.getBottomRightXY();
                    String label = labels.getLabel(obj.getPredictedClass());
                    xy1[0] = xy1[0] * w / gridWidth;
                    xy1[1] = xy1[1] * h / gridHeight;
                    xy2[0] = xy2[0] * w / gridWidth;
                    xy2[1] = xy2[1] * h / gridHeight;

                    centerX = (int) (xy1[0] + xy2[0]) / 2;
                    centerY = (int) (xy1[1] + xy2[1]) / 2;

                    rectangle(opencvMat, new Point((int) xy1[0], (int) xy1[1]), new Point((int) xy2[0], (int) xy2[1]), new Scalar(0, 255, 0, 0), 2, LINE_8, 0);
                    circle(opencvMat, new Point(centerX, centerY), 2, new Scalar(0, 255, 0, 0), -1, 0, 0);
                    putText(opencvMat, label, new Point((int) xy1[0] + 2, (int) xy2[1] - 2), FONT_HERSHEY_DUPLEX, 1, Scalar.GREEN);
                    centers.add(Nd4j.create(new float[]{(float) centerX, (float) centerY}));
                    people.add(Nd4j.create(new float[]{(float) xy1[0], (float) xy1[1], (float) xy2[0], (float) xy2[1]}));
                }
            }

            for (int i = 0; i < centers.size(); i++) {
                for (int j = 0; j < centers.size(); j++) {
                    if (i == j) {
                        continue;
                    }
                    double distance = euclideanDistance(centers.get(i), centers.get(j));
                    System.out.println("centre size for person " + i + " is " + centers);
                    System.out.println("centre get " + centers.get(i) + centers.get(j));
                    System.out.println("Distance Person " + i + " from Person " + j + " is " + distance);

                    if (distance < safeDistance && distance > 0) {
                        line(opencvMat, new Point(centers.get(i).getInt(0), centers.get(i).getInt(1)),
                                new Point(centers.get(j).getInt(0), centers.get(j).getInt(1)), Scalar.RED, 2, 1, 0);

                        violators.add(centers.get(i));
                        violators.add(centers.get(j));

                        int xmin = people.get(i).getInt(0);
                        int ymin = people.get(i).getInt(1);
                        int xmax = people.get(i).getInt(2);
                        int ymax = people.get(i).getInt(3);

                        rectangle(opencvMat, new Point(xmin, ymin), new Point(xmax, ymax), Scalar.RED, 2, LINE_8, 0);
                        circle(opencvMat, new Point(centers.get(i).getInt(0), centers.get(i).getInt(1)), 3, Scalar.RED, -1, 0, 0);
                    }
                }
            }
            putText(opencvMat, String.format("Number of people: %d", people.size()), new Point(10, 30), 4, 1.0, new Scalar(255, 0, 0, 0), 2, LINE_8, false);
            putText(opencvMat, String.format("Number of violators: %d", violators.size() / 2), new Point(10, 60), 4, 1.0, new Scalar(0, 0, 255, 0), 2, LINE_8, false);

            canvas.showImage(converter.convert(opencvMat));
            System.out.println(objs);

            KeyEvent t = canvas.waitKey(33);

            if ((t != null) && (t.getKeyCode() == KeyEvent.VK_Q)) {
                break;
            }

        }
        canvas.dispose();
    }

//
}
