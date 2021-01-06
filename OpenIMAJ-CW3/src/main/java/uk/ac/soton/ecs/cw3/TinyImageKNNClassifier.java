package uk.ac.soton.ecs.cw3;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.DoubleFVComparison;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.processing.resize.ResizeProcessor;
import org.openimaj.ml.annotation.basic.KNNAnnotator;

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
        System.out.println("Running Tiny Image KNN Classifier on full training set");
        //Load all images
        images = new VFSGroupDataset<>(CWD + "/OpenIMAJ-CW3/training", ImageUtilities.FIMAGE_READER);
        //Instantiate tiny extractor class
        TinyExtractor extractor = new TinyExtractor();

        // The annotator used to create the model.
        this.ann = KNNAnnotator.create(extractor, DoubleFVComparison.EUCLIDEAN, K);

        // Trains the model.
        System.out.println("Training Tiny Image KNN Annotator with full dataset for a robust model");
        this.ann.train(images);
    }

    /**
     * Method used to train the classifier will default to using the whole training data if parameters are not set using
     * setTestTrainSize(int TRAIN_SIZE, int TEST_SIZE) prior to training.
     */
    protected void initWithSplit() {
        System.out.println("Running Tiny Image KNN Classifier on split training set for testing purposes");

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

        // Trains the model with a split dataset
        System.out.println("Training Tiny Image KNN Annotator with split training dataset");
        this.ann.train(splits.getTrainingDataset());
    }

    /**
     * @return Instantiated KNNAnnotator
     * @throws Exception if annotator is null
     */
    public KNNAnnotator<FImage, String, DoubleFV> getAnnotator() throws Exception {
        if (this.ann != null) {
            return this.ann;
        } else {
            throw new Exception("Annotator is null / not been set");
        }
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

            // TODO mean centring and normalising makes KNN Worse!
            // output.processInplace(new MeanCenter());
            // DoubleFV feature = new DoubleFV(output.getDoublePixelVector());
            // feature.normaliseFV();

            return new DoubleFV(output.getDoublePixelVector());
        }
    }
}