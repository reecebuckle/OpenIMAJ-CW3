package uk.ac.soton.ecs.cw3;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.ml.annotation.ScoredAnnotation;

import java.io.IOException;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

public class Classifier {

    private int TRAIN_SIZE = 100; // The default training size
    private int TEST_SIZE = 0; // The default testing size
    protected final String CWD = System.getProperty("user.dir"); // The current working directory
    GroupedDataset<String, VFSListDataset<FImage>, FImage> images; // The images extracted to train/test the classifier
    GroupedRandomSplitter<String, FImage> splits; // The data splits for training/testing

    //TODO: Remove these abstract methods since they're different for each classifer??
    protected void classifyImages(String filename) throws IOException {
    }

    protected void trainClassifier() {
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
        images = new VFSGroupDataset<>(CWD + "/OpenIMAJ-CW3/training", ImageUtilities.FIMAGE_READER);
        splits = new GroupedRandomSplitter<>(images, TRAIN_SIZE, 0, TEST_SIZE);
    }
}