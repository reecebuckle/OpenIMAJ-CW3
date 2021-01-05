
package uk.ac.soton.ecs.cw3;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.feature.dense.gradient.dsift.ByteDSIFTKeypoint;
import org.openimaj.image.feature.dense.gradient.dsift.DenseSIFT;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.ml.annotation.bayes.NaiveBayesAnnotator;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;
import org.openimaj.util.pair.IntFloatPair;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Annotator based on a Naive Bayes Classifier. Uses a VectorNaiveBayesCategorizer as the actual classifier.
 **/
public class NaiveBayesClassifier extends Classifier {

    private final int STEP_SIZE;

    //parameters for Dense sift pyramid
    private final int BIN_SIZE;
    private final float E_THRESHOLD;

    private final int CLUSTERS; // Number of clusters in KNN
    private NaiveBayesAnnotator<FImage, String> ann;

    /**
     * Constructor used to set parameters of patch extractor and number of clusters of KNN.
     *
     * @param step_size   Number of steps between each patch in an image
     * @param bin_size    Size of the bin
     * @param e_threshold E threshold
     * @param clusters    Number of clusters used in KNN
     */
    public NaiveBayesClassifier(int step_size, int bin_size, float e_threshold, int clusters) {
        this.STEP_SIZE = step_size;
        this.BIN_SIZE = bin_size;
        this.E_THRESHOLD = e_threshold;
        this.CLUSTERS = clusters;

    }

    /**
     * Method used for initiating classifier on all training data.
     *
     * @throws FileSystemException Throws error if training folder is not present/suitable format
     */
    protected void init() throws FileSystemException {
        System.out.println("Running Naive Bayes Classifier on full dataset");
        //Load all images
        images = new VFSGroupDataset<>(CWD + "/OpenIMAJ-CW3/training", ImageUtilities.FIMAGE_READER);

        // Construct with the given step size (for both x and y) and bin size
        final DenseSIFT denseSIFT = new DenseSIFT(STEP_SIZE, BIN_SIZE);

        // Instantiate hard assigner with full training dataset
        HardAssigner<byte[], float[], IntFloatPair> assigner = trainQuantiser((VFSGroupDataset<FImage>) images, denseSIFT);

        // Utilise PHOW Extractor
        final FeatureExtractor<DoubleFV, FImage> extractor = new PHOWExtractor(assigner);

        // Use NaiveBayesAnnotator to construct and train classifier
        this.ann = new NaiveBayesAnnotator<>(extractor, NaiveBayesAnnotator.Mode.MAXIMUM_LIKELIHOOD);

        // Trains the model
        System.out.println("Training Naive Bayes Classifier with full dataset for a robust model");
        ann.train(images);

    }

    /**
     * Method used to train the classifier will default to using the whole training data if parameters are not set using
     * setTestTrainSize(int TRAIN_SIZE, int TEST_SIZE) prior to training.
     */
    protected void initWithSplit() {
        System.out.println("Running Naive Bayes Classifier with a split training set (for self testing purposes)");
        // Defaults to whole training set
        if (splits == null) {
            try {
                splitData();
            } catch (FileSystemException e) {
                e.printStackTrace();
            }
        }

        // Construct with the given step size (for both x and y) and bin size.
        final DenseSIFT denseSIFT = new DenseSIFT(STEP_SIZE, BIN_SIZE);

        // Instantiate hard assigner with a split training set
        HardAssigner<byte[], float[], IntFloatPair> assigner = trainQuantiser(splits.getTrainingDataset(), denseSIFT);

        // Utilise PHOW Extractor
        final FeatureExtractor<DoubleFV, FImage> extractor = new PHOWExtractor(assigner);

        // Use  to NaiveBayesAnnotator construct and train classifier
        this.ann = new NaiveBayesAnnotator<>(extractor, NaiveBayesAnnotator.Mode.MAXIMUM_LIKELIHOOD);

        // Trains the model.
        System.out.println("Training Naive Bayes Annotator with split training dataset");
        this.ann.train(splits.getTrainingDataset());
    }

    /**
     * @return Instantiated NaiveBayesAnnotator
     * @throws Exception if annotator is null
     */
    public NaiveBayesAnnotator<FImage, String> getAnnotator() throws Exception {
        if (this.ann != null) {
            return this.ann;
        } else {
            throw new Exception("Annotator is null / not been set");
        }
    }

    /**
     * TODO: Update description
     *
     * @param trainingDataset The dataset to be quantised (full training set to form robust model)
     * @param denseSIFT
     * @return HardAssigner that assigns the features to identifiers
     */
    public HardAssigner<byte[], float[], IntFloatPair> trainQuantiser(VFSGroupDataset<FImage> trainingDataset, final DenseSIFT denseSIFT) {
        System.out.println("Now Assigning features to images with HardAssigner");

        final AtomicReference<List<LocalFeatureList<ByteDSIFTKeypoint>>> allkeys = new AtomicReference<>(new ArrayList<>());

        for (Iterator<FImage> iterator = trainingDataset.iterator(); iterator.hasNext(); ) {
            FImage image = iterator.next();
            //Get sift features
            denseSIFT.analyseImage(image);
            allkeys.get().add(denseSIFT.getByteKeypoints());

        }

        // Cluster to learn vocabulary
        ByteKMeans km = ByteKMeans.createKDTreeEnsemble(CLUSTERS);
        final DataSource<byte[]> datasource = new LocalFeatureListDataSource<>(allkeys.get());

        ByteCentroidsResult result = km.cluster(datasource);

        return result.defaultHardAssigner();
    }

    /**
     * Overloaded method when splitting the input training set
     *
     * @param trainingDataset The dataset to be quantised (split training set)
     * @param denseSIFT
     * @return HardAssigner that assigns the features to identifiers
     */
    public HardAssigner<byte[], float[], IntFloatPair> trainQuantiser(GroupedDataset<String, ListDataset<FImage>, FImage> trainingDataset, final DenseSIFT denseSIFT) {
        System.out.println("Now Assigning features to images with HardAssigner");

        final AtomicReference<List<LocalFeatureList<ByteDSIFTKeypoint>>> allkeys = new AtomicReference<>(new ArrayList<>());

        for (Iterator<FImage> iterator = trainingDataset.iterator(); iterator.hasNext(); ) {
            FImage image = iterator.next();
            //Get sift features
            denseSIFT.analyseImage(image);
            allkeys.get().add(denseSIFT.getByteKeypoints());

        }

        // Cluster to learn vocabulary
        ByteKMeans km = ByteKMeans.createKDTreeEnsemble(CLUSTERS);
        final DataSource<byte[]> datasource = new LocalFeatureListDataSource<>(allkeys.get());

        ByteCentroidsResult result = km.cluster(datasource);

        return result.defaultHardAssigner();
    }

    public class PHOWExtractor implements FeatureExtractor<DoubleFV, FImage> {
        HardAssigner<byte[], float[], IntFloatPair> assigner;

        /**
         * Constructor used to set parameters of patch extractor and assigner
         *
         * @param assigner Assigns features to identifiers
         */
        public PHOWExtractor(HardAssigner<byte[], float[], IntFloatPair> assigner) {
            super();
            this.assigner = assigner;
        }

        /**
         * Method used to extract BOVW feature from an image
         *
         * @param image Image to extract features from
         * @return Bag-of-visual-words feature based on fixed size densely-sampled pixel patches
         */
        public DoubleFV extractFeature(FImage image) {

            final DenseSIFT denseSIFT = new DenseSIFT(STEP_SIZE, BIN_SIZE);

            denseSIFT.analyseImage(image);

            // Extracts basic (hard-assignment) bag of visual words (BOVW) representations of an image given a list of
            // local features and an HardAssigner
            BagOfVisualWords<byte[]> bovw = new BagOfVisualWords<>(assigner);

            //Use BlockSpatialAggregator together with BagOfVisualWords
            BlockSpatialAggregator<byte[], SparseIntFV> spatial =
                    new BlockSpatialAggregator<byte[], SparseIntFV>(bovw, 2, 2);

            return spatial.aggregate(denseSIFT.getByteKeypoints(E_THRESHOLD), image.getBounds()).normaliseFV();
        }
    }
}
