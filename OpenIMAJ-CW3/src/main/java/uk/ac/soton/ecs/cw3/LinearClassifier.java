package uk.ac.soton.ecs.cw3;

import de.bwaldvogel.liblinear.SolverType;
import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.FloatFV;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.LocalFeature;
import org.openimaj.feature.local.LocalFeatureImpl;
import org.openimaj.feature.local.SpatialLocation;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.image.pixel.sampling.RectangleSampler;
import org.openimaj.image.processing.algorithm.MeanCenter;
import org.openimaj.math.geometry.shape.Rectangle;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;
import org.openimaj.util.pair.IntFloatPair;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * LinearClassifier takes patches of pixels from images and performs vector quantisation to map each patch
 * to a visual word. A sample of these are clustered using K-Means to learn a vocabulary.
 */
public class LinearClassifier extends Classifier {

    private final int STEP; // Pixels between patches
    private final int SIZE; // Size of patches
    private final int CLUSTERS; // Number of clusters in KNN
    private LiblinearAnnotator<FImage, String> ann; // The annotator used for training the model

    /**
     * Constructor used to set parameters of patch extractor and number of clusters of KNN.
     *
     * @param step     Number of steps between each patch in an image
     * @param size     Size of the patches
     * @param clusters Number of clusters used in KNN
     */
    public LinearClassifier(int step, int size, int clusters) {
        this.STEP = step;
        this.SIZE = size;
        this.CLUSTERS = clusters;
    }

    /**
     * Method used for initiating classifier on all training data.
     *
     * @throws FileSystemException Throws error if training folder is not present/suitable format
     */
    protected void init() throws FileSystemException {
        System.out.println("Running Linear Classifier on full dataset");
        //Load all images
        images = new VFSGroupDataset<>(CWD + "/OpenIMAJ-CW3/training", ImageUtilities.FIMAGE_READER);

        // Instantiate patch extractor for densely-sampled pixel patches
        // The feature extractor used to preprocess each image.
        PatchExtractor patchExtractor = new PatchExtractor();

        // Instantiate hard assigner with full training dataset
        HardAssigner<float[], float[], IntFloatPair> assigner = trainQuantiser((VFSGroupDataset<FImage>) images, patchExtractor);

        // Use Bag of Visual Words Extractor
        FeatureExtractor<DoubleFV, FImage> extractor = new BOVWExtractor(patchExtractor, assigner);

        // Use LiblinearAnnotator to construct and train classifier
        this.ann = new LiblinearAnnotator<>(
                extractor, LiblinearAnnotator.Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);

        // Trains the model
        System.out.println("Training Linear Annotator with full dataset for a robust model");
        this.ann.train(images);
    }


    /**
     * Method used to train the classifier will default to using the whole training data if parameters are not set using
     * setTestTrainSize(int TRAIN_SIZE, int TEST_SIZE) prior to training.
     */
    protected void initWithSplit() {
        System.out.println("Running Linear Classifier with a split training set (for self testing purposes)");

        // Defaults to whole training set
        if (splits == null) {
            try {
                splitData();
            } catch (FileSystemException e) {
                e.printStackTrace();
            }
        }

        // Instantiate patch extractor for densely-sampled pixel patches
        // The feature extractor used to preprocess each image.
        PatchExtractor patchExtractor = new PatchExtractor();

        // Instantiate hard assigner
        HardAssigner<float[], float[], IntFloatPair> assigner = trainQuantiser(splits.getTrainingDataset(), patchExtractor);

        // Use Bag of Visual Words Extractor
        FeatureExtractor<DoubleFV, FImage> extractor = new BOVWExtractor(patchExtractor, assigner);

        // Use LiblinearAnnotator to construct and train classifier
        this.ann = new LiblinearAnnotator<>(
                extractor, LiblinearAnnotator.Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);

        // Trains the model.
        System.out.println("Training Linear Annotator with split training dataset");
        this.ann.train(splits.getTrainingDataset());
    }

    /**
     * @return Instantiated LiblinearAnnotator
     * @throws Exception if annotator is null
     */
    public LiblinearAnnotator<FImage, String> getAnnotator() throws Exception {
        if (this.ann != null) {
            return this.ann;
        } else {
            throw new Exception("Annotator is null / not been set");
        }
    }

    /**
     * Method used for testing the accuracy of the linear classifier specifically
     *
     * @return Accuracy of the model
     */
    protected Double testLinearClassifier() {
        System.out.println("Testing accuracy of the Linear Classifier");
        Double n_correct = 0.0;
        int n = splits.getTestDataset().size();

        // Iterate over test set and classify each image
        // Compare against actual label
        for (Map.Entry<String, ListDataset<FImage>> set : splits.getTestDataset().entrySet()) {
            for (FImage image : set.getValue()) {
                String predicted = ann.classify(image).getPredictedClasses().toString().replaceAll("\\[|\\]", "");
                if (predicted.equals(set.getKey())) n_correct += 1;
            }
        }

        return n_correct / n;
    }

    /**
     * Method that quantises images using fixed size densely-sampled pixel patches
     * The vectors are then clustered using K-Means to learn a vocabularyOverloaded method when training the full training set (not splitting data)
     *
     * @param fullTrainingDataset The dataset to be quantised
     * @param extractor           The extractor used to extract patches
     * @return HardAssigner that assigns the features to identifiers
     * @override
     */
    public HardAssigner<float[], float[], IntFloatPair> trainQuantiser(VFSGroupDataset<FImage> fullTrainingDataset, PatchExtractor extractor) {
        System.out.println("Now Assigning features to images with HardAssigner");

        List<float[]> allkeys = new ArrayList<float[]>();

        GroupedRandomSplitter<String, FImage> splitter = new GroupedRandomSplitter<String, FImage>(fullTrainingDataset,10,0,0);

        // Iterate through training dataset extracting the patches
        for (Map.Entry<String, ListDataset<FImage>> images : splitter.getTrainingDataset().entrySet()) {
            for (FImage image : images.getValue()) {
                List<LocalFeature<SpatialLocation, FloatFV>> sampleList = extractor.extract(image, STEP, SIZE);

                for (LocalFeature<SpatialLocation, FloatFV> localFeature : sampleList) {
                    allkeys.add(localFeature.getFeatureVector().values);
                }
            }
        }
        System.out.println("Finished etracting images to assign features");

        // Cluster to learn vocabulary
        FloatKMeans km = FloatKMeans.createKDTreeEnsemble(CLUSTERS);
        float[][] datasource = allkeys.toArray(new float[][]{});
        FloatCentroidsResult result = km.cluster(datasource);
        System.out.println("Finished Clustering");

        return result.defaultHardAssigner();
    }

    /**
     * Overloaded method when training the a split dataset
     *
     * @param trainingDataset The dataset to be quantised
     * @param extractor       The extractor used to extract patches
     * @return HardAssigner that assigns the features to identifiers
     */
    public HardAssigner<float[], float[], IntFloatPair> trainQuantiser(GroupedDataset<String, ListDataset<FImage>, FImage> trainingDataset, PatchExtractor extractor) {
        System.out.println("Now Assigning features to images with HardAssigner");

        List<float[]> allkeys = new ArrayList<float[]>();

        // Iterate through training dataset extracting the patches
        for (Map.Entry<String, ListDataset<FImage>> images : trainingDataset.entrySet()) {
            for (FImage image : images.getValue()) {
                List<LocalFeature<SpatialLocation, FloatFV>> sampleList = extractor.extract(image, STEP, SIZE);

                for (LocalFeature<SpatialLocation, FloatFV> localFeature : sampleList) {
                    allkeys.add(localFeature.getFeatureVector().values);
                }
            }
        }
        System.out.println("Finished extracting images to assign features");

        // Cluster to learn vocabulary
        FloatKMeans km = FloatKMeans.createKDTreeEnsemble(CLUSTERS);
        float[][] datasource = allkeys.toArray(new float[][]{});
        FloatCentroidsResult result = km.cluster(datasource);
        System.out.println("Finished Clustering");

        return result.defaultHardAssigner();
    }


    public class PatchExtractor {

        /**
         * Method used to get features extracted from patches of an image
         *
         * @param image Image to extract the patches from
         * @param step  Distance between patches
         * @param size  Size of the patches
         * @return List of the patch features
         */
        public List<LocalFeature<SpatialLocation, FloatFV>> extract(FImage image, int step, int size) {

            // Sampler used to return x,y coordinates of patches
            RectangleSampler sampler = new RectangleSampler(image, step, step, size, size);
            List<LocalFeature<SpatialLocation, FloatFV>> features = new ArrayList<LocalFeature<SpatialLocation, FloatFV>>();

            // Iterate over x,y coordinates of patches and calculate features
            for (Rectangle rectangle : sampler) {
                FImage patch = image.extractROI(rectangle);
                patch.processInplace(new MeanCenter());
                patch.normalise();
                FloatFV feature = new FloatFV(patch.getFloatPixelVector());
                LocalFeature<SpatialLocation, FloatFV> localFeature = new LocalFeatureImpl<SpatialLocation, FloatFV>(
                        new SpatialLocation(rectangle.x, rectangle.y), feature);
                features.add(localFeature);
            }

            return features;
        }
    }


    public class BOVWExtractor implements FeatureExtractor<DoubleFV, FImage> {
        PatchExtractor patchExtractor;
        HardAssigner<float[], float[], IntFloatPair> assigner;

        /**
         * Constructor used to set parameters of patch extractor and assigner
         *
         * @param patchExtractor Extracts patch features from images
         * @param assigner       Assigns features to identifiers
         */
        public BOVWExtractor(PatchExtractor patchExtractor, HardAssigner<float[], float[], IntFloatPair> assigner) {
            this.patchExtractor = patchExtractor;
            this.assigner = assigner;
        }

        /**
         * Method used to extract BOVW feature from an image
         *
         * @param image Image to extract features from
         * @return Bag-of-visual-words feature based on fixed size densely-sampled pixel patches
         */
        public DoubleFV extractFeature(FImage image) {
            // Extracts basic (hard-assignment) bag of visual words (BOVW) representations of an image given a list of
            // local features and an HardAssigner
            BagOfVisualWords<float[]> bovw = new BagOfVisualWords<float[]>(assigner);

            //Use BlockSpatialAggregator together with BagOfVisualWords
            BlockSpatialAggregator<float[], SparseIntFV> spatial =
                    new BlockSpatialAggregator<float[], SparseIntFV>(bovw, 2, 2);

            return spatial.aggregate(patchExtractor.extract(image, STEP, SIZE), image.getBounds()).normaliseFV();
        }
    }

}