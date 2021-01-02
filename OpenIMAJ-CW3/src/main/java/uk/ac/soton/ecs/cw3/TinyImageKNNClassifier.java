package uk.ac.soton.ecs.cw3;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.DoubleFVComparison;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.processing.resize.ResizeProcessor;
import org.openimaj.ml.annotation.ScoredAnnotation;
import org.openimaj.ml.annotation.basic.KNNAnnotator;


import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;
import java.util.Map;

/**
 * TinyImageKNNClassifier uses K Nearest Neighbours in order to classify a set of images given a test set.
 * To achieve this the image is scaled down to a set pixel size. A feature vector is then extracted by
 * concatenating each row of the image into a 1D vector. Each vector is then used to train the classifier.
 */

public class TinyImageKNNClassifier extends Classifier {

    private final int SIZE; // The size of the image to scale
    private final int K; // The number of neighbours
    private KNNAnnotator<FImage, String, DoubleFV> ann; // The annotator used for training the model

    /**
     * Constructor used to set the resized image size and the number of neighbours to use in the model.
     *
     * @param size The size of the width and height used to resize the image.
     * @param k    The number of neighbours used to classify new images.
     */
    public TinyImageKNNClassifier(int size, int k) {
        this.SIZE = size;
        this.K = k;

    }

    /**
     * Method used for initiating classifier on all training data.
     *
     * @throws FileSystemException Does what is says on the tin
     */
    protected void init() throws FileSystemException {
        //Load all images
        images = new VFSGroupDataset<>(CWD + "/OpenIMAJ-CW3/training", ImageUtilities.FIMAGE_READER);
        //Instantiate tiny extractor class
        TinyExtractor extractor = new TinyExtractor();

        // The annotator used to create the model.
        this.ann = KNNAnnotator.create(extractor, DoubleFVComparison.EUCLIDEAN, K);
        // Trains the model.
        this.ann.train(images);
    }

    /**
     * Method used to classify new images that outputs a text file containing the results.
     *
     * @param filename The name of the file to output the results to.
     * @throws IOException Throws IO exception if test data is not present.
     */
    @Override
    protected void classifyImages(String filename) throws IOException {

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
     * Method used to train the classifier will default to using the whole training data if parameters are not set using
     * setTestTrainSize(int TRAIN_SIZE, int TEST_SIZE) prior to training.
     */
    @Override
    protected void trainClassifier() {

        // Defaults to whole training set
        if (splits == null) {
            try {
                splitData();
            } catch (FileSystemException e) {
                e.printStackTrace();
            }
        }

        // The feature extractor used to preprocess each image.
        TinyExtractor extractor = new TinyExtractor();

        // The annotator used to create the model.
        this.ann = KNNAnnotator.create(extractor, DoubleFVComparison.EUCLIDEAN, K);
        // Trains the model.
        this.ann.train(splits.getTrainingDataset());
    }

    /**
     * Method used for testing the accuracy of the classifier and returns the summary report.
     *
     * @return The summary report of the classifier using the train/test split
     */
    protected String testClassifier() {
        // The evaluator that is passed the test split to evaluate the models performance
        ClassificationEvaluator<CMResult<String>, String, FImage> eval =
                new ClassificationEvaluator<>(
                        ann, splits.getTestDataset(), new CMAnalyser<FImage, String>(CMAnalyser.Strategy.SINGLE));

        // Evaluates the results
        Map<FImage, ClassificationResult<String>> guesses = eval.evaluate();
        CMResult<String> result = eval.analyse(guesses);

        return result.getSummaryReport();
    }


    private class TinyExtractor implements FeatureExtractor<DoubleFV, FImage> {
        /**
         * Class used to extract the features from the images. This is accomplished by first detecting if the image
         * is rectangular and cropping the image to a square about its center if needed. Then the image is resized using
         * using SIZE to determine how big the image is. Finally the image is converted into a 1D column vector by
         * concatenating each row of pixles.
         *
         * @param image The image to be preprocessed to have its features extracted.
         * @return The extracted feature vector used in the classifier.
         */
        public DoubleFV extractFeature(FImage image) {

            int width = image.getWidth(), height = image.getHeight();

            FImage output = image.clone();
            // Detects if the image needs to be cropped or not.
            if (width != height) {
                int size = Math.min(width, height);
                output = output.extractCenter(size, size);
            }

            // Resizes the image
            output.processInplace(new ResizeProcessor(SIZE, SIZE));

            // TODO CAN SOMEONE EXPLAIN WHY THIS CODE IS COMMENTED OUT, OR REMOVE IT
            // Subtracts the mean pixel value from the entire image
            // output.processInplace(new MeanCenter());
            // Creates the feature vector (1D vector)
            // DoubleFV feature = new DoubleFV(output.getDoublePixelVector());
            // Normalises the feature vector to make each pixel unit length
            // feature.normaliseFV();

            return new DoubleFV(output.getDoublePixelVector());
        }
    }
}