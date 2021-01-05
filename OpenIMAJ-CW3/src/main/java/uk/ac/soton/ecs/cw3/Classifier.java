package uk.ac.soton.ecs.cw3;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.ml.annotation.Annotator;
import org.openimaj.ml.annotation.ScoredAnnotation;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Map;

public class Classifier {

    private int TRAIN_SIZE = 100; // The default training size
    private int TEST_SIZE = 0; // The default testing size
    protected final String CWD = System.getProperty("user.dir"); // The current working directory
    GroupedDataset<String, VFSListDataset<FImage>, FImage> images; // The images extracted to train/test the classifier
    GroupedRandomSplitter<String, FImage> splits; // The data splits for training/testing

    /**
     * Method used to classify new images that outputs a text file containing the results.
     *
     * @param filename The name of the file to output the results to.
     * @throws IOException Throws IO exception if test data is not present.
     */

    protected void classifyImages(String filename, Annotator ann) throws IOException {
        System.out.println("Now Classifying Images");
        // Used to write the results to
        FileWriter fileWriter = new FileWriter(filename);

        // Opens directory to get filenames. Then gets the filenames and sorts them numerically
        File test = new File(CWD + "\\OpenIMAJ-CW3\\testing");
        String[] filenames = test.list();
        sortFilenames(filenames);

        // Loops through files, then classifies them using the annotator and writes the results to the file.
        if (filenames != null) {
            for (String file : filenames) {
                FImage image = ImageUtilities.readF(new File(".\\OpenIMAJ-CW3\\testing\\" + file));
                List<ScoredAnnotation<String>> result = ann.annotate(image);
                fileWriter.write(String.format("%s %s\n", file, getClassification(result)));
            }
        }
        fileWriter.close();
    }

    /**
     * Method used to adjust the training and testing sizes used to split the data for validation.
     *
     * @param TRAIN_SIZE The number of elements used to create the training set.
     * @param TEST_SIZE  The number of elements used to create the testing set.
     */
    protected void setTestTrainSize(int TRAIN_SIZE, int TEST_SIZE) {
        this.TRAIN_SIZE = TRAIN_SIZE;
        this.TEST_SIZE = TEST_SIZE;
    }

    /**
     * Method for sorting filenames numerically so that they appear in order when writing results.
     *
     * @param arr String[] containing unsorted filenames
     */
    protected void sortFilenames(String[] arr) {
        // Compares the first numerical part of the filename prior to the extension and sorts array in place.
        Arrays.sort(arr, new Comparator<String>() {
            @Override
            public int compare(String o1, String o2) {
                Integer a = Integer.parseInt(o1.split("\\.")[0]);
                Integer b = Integer.parseInt(o2.split("\\.")[0]);
                return a.compareTo(b);
            }
        });
    }

    /**
     * Method for retrieving the image class with the highest confidence score
     *
     * @param result The list of annotated results containing confidence to multiple images classes determined by their
     *               neighbouring images.
     * @return The image class with the highest confidence score
     */
    protected String getClassification(List<ScoredAnnotation<String>> result) {

        // Compares all confidence score and determines the maximum score and corresponding image class
        float min = 0f;
        String group = null;
        for (ScoredAnnotation<String> annotation : result) {
            if (annotation.confidence > min) {
                min = annotation.confidence;
                group = annotation.annotation;
            }
        }

        return group;
    }

    /**
     * Method used to split the data into training and testing sets. This is done using the classes TRAIN_SIZE and
     * TEST_SIZE.
     *
     * @throws FileSystemException Throws an exception if the training set is not in the correct folder.
     */
    protected void splitData() throws FileSystemException {
        System.out.println("Splitting input training set by : " + TRAIN_SIZE + " / " + TEST_SIZE);
        images = new VFSGroupDataset<>(CWD + "/OpenIMAJ-CW3/training", ImageUtilities.FIMAGE_READER);
        splits = new GroupedRandomSplitter<>(images, TRAIN_SIZE, 0, TEST_SIZE);
    }

    /**
     * Method used for testing the accuracy of the classifier and returns the summary report.
     *
     * @return The summary report of the classifier using the train/test split
     */
    protected String testClassifier(Annotator ann) {
        System.out.println("Testing accuracy of Classifier");
        // The evaluator that is passed the test split to evaluate the models performance
        ClassificationEvaluator<CMResult<String>, String, FImage> eval =
                new ClassificationEvaluator<>(
                        ann, splits.getTestDataset(), new CMAnalyser<FImage, String>(CMAnalyser.Strategy.SINGLE));

        // Evaluates the results
        Map<FImage, ClassificationResult<String>> guesses = eval.evaluate();
        CMResult<String> result = eval.analyse(guesses);

        //print out report
        System.out.println(result.getDetailReport());

        return result.getSummaryReport();
    }
}