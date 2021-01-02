package uk.ac.soton.ecs.cw3;

import java.io.FileWriter;
import java.io.IOException;

/**
 * Hi team, hope you guys all got here!
 *
 */
public class App {

    public static void main( String[] args ) {

        try {

            LinearClassifier linearClassifier = new LinearClassifier(4,8,500);
            linearClassifier.trainClassifier();
            linearClassifier.classifyImages("Linear Classifier Results.txt");

            /**
            TinyImageKNNClassifier tinyImageKNNClassifier = new TinyImageKNNClassifier(16, 19);
            tinyImageKNNClassifier.trainClassifier();
            tinyImageKNNClassifier.classifyImages("Tiny Image Classifier Results.txt");

//            testTinyImageKNN();
             */

        } catch (IOException ioException) {
            ioException.printStackTrace();
        }


    }

    /**
     * Method used for testing the tinyImageKNNClassifier to tune k and or small image size.
     * @throws IOException Does what it says on the tin
     */
    public static void testTinyImageKNN() throws IOException {

        FileWriter fileWriter = new FileWriter("Tiny Image Classifier Tuning.txt");

        for (int k = 19; k < 20; k++) {

            int n = 20;
            double[] acc_avg = new double[n], err_avg = new double[n];

            fileWriter.write("K, accuracy, error\n");
            for (int i = 0; i < n; i++) {
                TinyImageKNNClassifier tinyImageKNNClassifier = new TinyImageKNNClassifier(16, k);

                // Load the dataset
                tinyImageKNNClassifier.setTestTrainSize(80, 20);
                tinyImageKNNClassifier.splitData();


                tinyImageKNNClassifier.trainClassifier();

                String results = tinyImageKNNClassifier.testClassifier();
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
