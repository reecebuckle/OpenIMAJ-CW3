package uk.ac.soton.ecs.cw3;

import org.apache.commons.vfs2.FileSystemException;
import org.apache.tools.ant.DynamicElementNS;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;

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

            //runTinyImageKNNClassifier();
            //testTinyImageKNNClassifier();

            //runLinearClassifier();
            //testLinearClassifier();

            //runNaiveBayesClassifier();
            testNaiveBayesClassifier();

            //runCNN();

            //Output running time
            long endTime = System.currentTimeMillis();
            NumberFormat formatter = new DecimalFormat("#0.00000");
            System.out.println("Execution time is " + formatter.format((endTime - startTime) / 1000d) + " seconds");
        } catch (IOException ioException) {
            ioException.printStackTrace();
        }
    }

    /**
     * Method for running DenseSIFTClassifier
     * Instantiates a DenseSIFTClassifier with <Clusters, siftStep, siftFeatures>
     *     
     * @// TODO: 05/01/2021 Try increasing clusters (600?), reducing siftStep (3?), increase siftFeatures (2000?, 1500?)
     *     
     * @throws FileSystemException
     */
    public static void runDenseSIFT() throws FileSystemException {
        GroupedDataset<String, VFSListDataset<FImage>, FImage> images =
                new VFSGroupDataset<>(System.getProperty("user.dir") + "/OpenIMAJ-CW3/training", ImageUtilities.FIMAGE_READER);

        GroupedRandomSplitter<String, FImage> splitter = new GroupedRandomSplitter<String, FImage>(images, 80, 0, 20);

        DenseSIFTClassifer classifier = new DenseSIFTClassifer(300, 5,1000);
        LiblinearAnnotator<FImage,String> ann = classifier.constructAnnotator(splitter.getTrainingDataset());
        classifier.getReport(ann, splitter.getTestDataset());
    }


    /**
     * Runs Max's implementation of Convolutional Neural Network
     *
     * @throws IOException
     * @throws InterruptedException
     */
    public static void runCNN() throws IOException, InterruptedException {
            CNN cnn = new CNN();
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
     * Method used for self testing of the Linear Classifier by splitting the dataset 90/10
     *
     * @throws Exception (can be IO or error with annotator)
     */
    public static void testLinearClassifier() throws Exception {

        LinearClassifier linearClassifier = new LinearClassifier(5, 8, 500); //TODO SET THIS AS 500 CLUSTERS, 5 IS JUST FOR QUICK RUNNING
        // Set training / testing split and split
        linearClassifier.setTestTrainSize(80, 20);
        linearClassifier.splitData();
        // Initialise classifier with a split dataset
        linearClassifier.initWithSplit();
        // Test Classifier with Linear Classifier
        linearClassifier.testClassifier(linearClassifier.getAnnotator());
    }

    /**
     * Method used for running Naive Bayes Classifier with PHOW Extractor
     *
     * @throws Exception (can be IO or error with annotator)
     */
    public static void runNaiveBayesClassifier() throws Exception {
        NaiveBayesClassifier naiveBayesClassifier = new NaiveBayesClassifier(4, 8, 0.015f, 5);
        naiveBayesClassifier.init();
        naiveBayesClassifier.classifyImages("naive_bayes_run3.txt", naiveBayesClassifier.getAnnotator());
    }


    /**
     * Method used for testing Naives Bayes Classifier with PHOW Extractor
     *
     * @throws Exception (can be IO or error with annotator)
     */
    public static void testNaiveBayesClassifier() throws Exception {
        //NaiveBayesClassifier naiveBayesClassifier = new NaiveBayesClassifier(2, 4, 0.015f, 25); //TODO try changing these params - results in discord - returns 61%
        NaiveBayesClassifier naiveBayesClassifier = new NaiveBayesClassifier(2, 8, 0.015f, 500); //TODO Default params (written by Turgut)
        //Set training / testing split and split
        naiveBayesClassifier.setTestTrainSize(90, 10);
        naiveBayesClassifier.splitData();
        //Initialise classifier with a split dataset
        naiveBayesClassifier.initWithSplit();
        //Test Classifier with NaiveBayes Classifier
        naiveBayesClassifier.testClassifier(naiveBayesClassifier.getAnnotator());
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
     * Method used for testing the tinyImageKNNClassifier to tune k and or small image size.
     *
     * @throws Exception (can be IO or error with annotator)
     */
    public static void testTinyImageKNN() throws Exception {

        FileWriter fileWriter = new FileWriter("Tiny Image Classifier Tuning.csv");
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
