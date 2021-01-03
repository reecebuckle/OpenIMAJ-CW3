package uk.ac.soton.ecs.cw3;

import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.image.FImage;
import org.openimaj.ml.annotation.basic.KNNAnnotator;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;

import javax.sound.sampled.Line;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.text.NumberFormat;

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
            long startTime = System.currentTimeMillis();

            //TODO Uncomment to run
            runTinyImageKNNClassifier();
            //runLinearClassifier();

            //Output running time
            long endTime = System.currentTimeMillis();
            NumberFormat formatter = new DecimalFormat("#0.00000");
            System.out.print("Execution time is " + formatter.format((endTime - startTime) / 1000d) + " seconds");
        } catch (IOException ioException) {
            ioException.printStackTrace();
        }
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
        tinyImageKNNClassifier.classifyImages("run1.txt", tinyImageKNNClassifier.getAnnotator());
    }

    /**
     * Method used for running the Linear Classifier (Task two)
     * Instantiates a Linear Classifier with <Step amount, Size, Cluster amount>
     *
     * @throws Exception (can be IO or error with annotator)
     */
    public static void runLinearClassifier() throws Exception {

        //Instantiate a Linear Classifier with <Step amount, Size, Cluster amount>
        LinearClassifier linearClassifier = new LinearClassifier(4, 8, 500); //TODO SET THIS AS 500 CLUSTERS, 5 IS JUST FOR QUICK RUNNING
        //Train classifier with full dataset
        linearClassifier.init();
        //Classify images with LibLinear Annotator
        linearClassifier.classifyImages("run2.txt", linearClassifier.getAnnotator());
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

    /**
     * Method used for self testing of the Linear Classifier by splitting the dataset 90/10
     *
     * @throws Exception (can be IO or error with annotator)
     */
    public static void testLinearClassifier() throws Exception {

        LinearClassifier linearClassifier = new LinearClassifier(4, 8, 500); //TODO SET THIS AS 500 CLUSTERS, 5 IS JUST FOR QUICK RUNNING
        // Set training / testing split and split
        linearClassifier.setTestTrainSize(90, 10);
        linearClassifier.splitData();
        // Initialise classifier with a split dataset
        linearClassifier.initWithSplit();
        linearClassifier.testClassifier(linearClassifier.getAnnotator());

    }
}
