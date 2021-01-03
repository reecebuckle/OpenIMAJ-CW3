package uk.ac.soton.ecs.cw3;

import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.image.FImage;
import org.openimaj.ml.annotation.basic.KNNAnnotator;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;

import javax.sound.sampled.Line;
import java.io.FileWriter;
import java.io.IOException;

/**
 * Hi team, hope you guys all got here!
 */
public class App {

    /**
     * Used to run the classifiers respectively
     *
     * @param args
     */
    public static void main(String[] args) throws Exception {

        try {
            //Output running time
            long startTime = System.nanoTime();

            //runTinyImageKNNClassifier(); //TODO Uncomment to run
            runLinearClassifier();

            long stopTime = System.nanoTime();
            System.out.println("Total running time: " + (stopTime - startTime));
        } catch (IOException ioException) {
            ioException.printStackTrace();
        }
    }

    /**
     * Method used for running the Linear Classifier (Task two)
     * Instantiates a Linear Classifier with <Step amount, Size, Cluster amount>
     *
     * @throws Exception (can be IO or error with annotator)
     */
    public static void runLinearClassifier() throws Exception {

        //Instantiate a Linear Classifier with <Step amount, Size, Cluster amount>
        LinearClassifier linearClassifier = new LinearClassifier(4, 8, 2); //TODO SET THIS AS 500 CLUSTERS, 5 IS JUST FOR QUICK RUNNING

        //Train classifier with full dataset
        linearClassifier.init();

        //Classify images with LibLinear Annotator
        linearClassifier.classifyImages("Linear Classifier Results.txt", linearClassifier.getAnnotator());
    }

    /**
     * Method used for running the Tiny Image KNN Classifier
     * InstantiateS tiny image KNN classifier of <size 16, K Clusters>
     *
     * @throws Exception (can be IO or error with annotator)
     */
    public static void runTinyImageKNNClassifier() throws Exception {

        //Instantiate tiny image KNN classifier of <size 16, K Clusters>
        TinyImageKNNClassifier tinyImageKNNClassifier = new TinyImageKNNClassifier(16, 18);

        //Train classifier with full dataset
        tinyImageKNNClassifier.init();

        //Classify with KNN Annotator
        tinyImageKNNClassifier.classifyImages("Tiny Image Classifier Results.txt", tinyImageKNNClassifier.getAnnotator());
    }

    /**
     * TODO: Write Method used for testing the LibLinear Classifier
     *
     * @throws Exception (can be IO or error with annotator)
     */
    public static void testLinearClassifier() throws Exception {

        LinearClassifier linearClassifier = new LinearClassifier(4, 8, 5); //TODO SET THIS AS 500 CLUSTERS, 5 IS JUST FOR QUICK RUNNING
        LiblinearAnnotator annotator = linearClassifier.getAnnotator();
        linearClassifier.setTestTrainSize(90, 10);
        linearClassifier.splitData();
        linearClassifier.initWithSplit();
        linearClassifier.classifyImages("Linear Classifier Results.txt", annotator);
    }

    /**
     * Method used for testing the tinyImageKNNClassifier to tune k and or small image size.
     *
     * @throws Exception (can be IO or error with annotator)
     */
    public static void testTinyImageKNN() throws Exception {

        FileWriter fileWriter = new FileWriter("Tiny Image Classifier Tuning 70-30.csv");
        fileWriter.write("K, accuracy, error\n");

        for (int k = 1; k <= 50; k++) {

            System.out.println(String.format("Calculating accuracy/error for k-neighbours: %d", k));

            int n = 20;
            double[] acc_avg = new double[n], err_avg = new double[n];

            for (int i = 0; i < n; i++) {
                TinyImageKNNClassifier tinyImageKNNClassifier = new TinyImageKNNClassifier(16, k);

                // Set training / testing split and split
                tinyImageKNNClassifier.setTestTrainSize(90, 10);
                tinyImageKNNClassifier.splitData();
                // Initialise classifier with a split dataset
                tinyImageKNNClassifier.initWithSplit();

                // Test Classifier with KNN Annotator
                String results = tinyImageKNNClassifier.testClassifier(tinyImageKNNClassifier.getAnnotator());

                // Format results for csv analysis
                String accuracy = results.split("\n")[0].split(": ")[1].replaceAll("\\s+", "");
                String error = results.split("\n")[1].split(": ")[1].replaceAll("\\s+", "");
                acc_avg[i] = Double.parseDouble(accuracy);
                err_avg[i] = Double.parseDouble(error);
            }

            double acc_sum = 0;
            double err_sum = 0;
            for (int d = 0; d < n; d++) {
                acc_sum += acc_avg[d];
                err_sum += err_avg[d];
            }

            double accuracy = acc_sum / acc_avg.length;
            double error = err_sum / acc_avg.length;

            fileWriter.write(String.format("%d, %f, %f\n", k, accuracy, error));
        }

        fileWriter.close();
    }
}
