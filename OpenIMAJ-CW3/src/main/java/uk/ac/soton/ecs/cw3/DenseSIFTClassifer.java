package uk.ac.soton.ecs.cw3;

import de.bwaldvogel.liblinear.SolverType;
import org.apache.commons.vfs2.FileSystemException;
import org.opencv.xfeatures2d.SIFT;
import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.*;
import org.openimaj.experiment.dataset.sampling.GroupedUniformRandomisedSampler;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.feature.dense.gradient.dsift.ByteDSIFTKeypoint;
import org.openimaj.image.feature.dense.gradient.dsift.DenseSIFT;
import org.openimaj.image.feature.dense.gradient.dsift.PyramidDenseSIFT;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.PyramidSpatialAggregator;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;
import org.openimaj.ml.kernel.HomogeneousKernelMap;
import org.openimaj.util.pair.IntFloatPair;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * DenseSIFTClassifier takes SIFT features from images and performs vector quantisation to map each feature
 * into visual words. Spatial histograms of the visual word occurrences are build using K-means clustering.
 */
public class DenseSIFTClassifer extends Classifier {

    private final int CLUSTERS;
    private final int SIFTSTEP;
    private final int SIFTFEATURES;

    /**
     * Constructor used to set various
     * @param clusters number of visual words in the vocabulary
     * @param siftStep step size for the SIFT features
     * @param siftFeatures max number of SIFT features taken from all the images
     */
    public DenseSIFTClassifer(int clusters, int siftStep, int siftFeatures) {
        this.CLUSTERS = clusters;
        this.SIFTSTEP = siftStep;
        this.SIFTFEATURES = siftFeatures;
    }

    /**
     * Method used for training and returning an annotator
     * @param dataset the training dataset on which to train the annotator
     * @return the annotator for predicting the label of new images
     */
    public LiblinearAnnotator<FImage,String>constructAnnotator(GroupedDataset<String, ListDataset<FImage>, FImage> dataset) {

        // Construct feature extractor
        DenseSIFT dsift = new DenseSIFT(SIFTSTEP, 7);

        // Apply feature extractor to windows of size 7
        PyramidDenseSIFT<FImage> pdsift = new PyramidDenseSIFT<FImage>(dsift, 6f, 7);

        System.out.println("Obtaining SIFT features from images and clustering.");
        // Perform K-Means clustering on features
        HardAssigner<byte[], float[], IntFloatPair> assigner =
                trainQuantiser(GroupedUniformRandomisedSampler.sample(dataset, 30), pdsift);

        System.out.println("Running BoVW feature extractor...");
        // Feature extractor to train classifier
        FeatureExtractor<DoubleFV, FImage> extractor = new PHOWExtractor(pdsift, assigner);

        HomogeneousKernelMap map = new HomogeneousKernelMap(
                HomogeneousKernelMap.KernelType.Chi2, HomogeneousKernelMap.WindowType.Rectangular);

        System.out.println("Creating HomogeneuosKernelMap...");
        extractor = map.createWrappedExtractor(extractor);

        System.out.println("Training the linear classifier...");
        // Construct and train linear classifier
        LiblinearAnnotator<FImage, String> ann = new LiblinearAnnotator<FImage, String>(
                extractor, LiblinearAnnotator.Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);

        ann.train(dataset);

        return ann;
    }

    /**
     * Method used to display detailed reports on the accuracy of a model
     * @param ann the model to test
     * @param dataset the dataset on which to test the model
     */
    public void getReport(LiblinearAnnotator<FImage,String> ann, GroupedDataset<String, ListDataset<FImage>, FImage> dataset) {
        // Obtain accuracy of classifier
        ClassificationEvaluator<CMResult<String>, String, FImage> eval =
                new ClassificationEvaluator<CMResult<String>, String, FImage>(
                        ann, dataset, new CMAnalyser<FImage,String>(CMAnalyser.Strategy.SINGLE));

        Map<FImage, ClassificationResult<String>> guesses = eval.evaluate();

        CMResult<String> result = eval.analyse(guesses);

        System.out.println(result.getDetailReport());
        System.out.println(result.getSummaryReport());
    }

    /**
     * Method to perform K-Means clustering on a sample of SIFT features
     * @param sample the sample of images
     * @param pdsift the pyramid sift feature extractor
     * @return
     */
    private HardAssigner<byte[], float[], IntFloatPair> trainQuantiser(
            GroupedDataset<String, ListDataset<FImage>,FImage> sample, PyramidDenseSIFT<FImage> pdsift) {
        List<LocalFeatureList<ByteDSIFTKeypoint>> allkeys = new ArrayList<LocalFeatureList<ByteDSIFTKeypoint>>();

        for (FImage img : sample) {
            pdsift.analyseImage(img);
            allkeys.add(pdsift.getByteKeypoints(0.005f));
        }

        // Randomize keys
        Collections.shuffle(allkeys);

        // Take first subset of dense SIFT features
        if (allkeys.size() > SIFTFEATURES)
            allkeys = allkeys.subList(0, SIFTFEATURES);

        System.out.println("Clustering...");
        // Cluster the features into separate classes
        ByteKMeans km = ByteKMeans.createKDTreeEnsemble(CLUSTERS);
        DataSource<byte[]> datasource = new LocalFeatureListDataSource<ByteDSIFTKeypoint, byte[]>(allkeys);
        ByteCentroidsResult result = km.cluster(datasource);

        return result.defaultHardAssigner();
    }


    private class PHOWExtractor implements FeatureExtractor<DoubleFV, FImage> {
        PyramidDenseSIFT<FImage> pdsift;
        HardAssigner<byte[], float[], IntFloatPair> assigner;

        public PHOWExtractor(PyramidDenseSIFT<FImage> pdsift, HardAssigner<byte[], float[], IntFloatPair> assigner) {
            this.pdsift = pdsift;
            this.assigner = assigner;
        }

        @Override
        /**
         * Extracts the BoVW features for an image
         * @param image the image to extract features from
         */
        public DoubleFV extractFeature(FImage image) {
            pdsift.analyseImage(image);

            /**
             * Compute 4 histograms across the image
             * BagOfVisualWords uses HardAssigner to assign each Dense SIFT feature to a
             * visual word and compute histogram
             * */
            BagOfVisualWords<byte[]> bovw = new BagOfVisualWords<byte[]>(assigner);

            PyramidSpatialAggregator<byte[], SparseIntFV> spatial = new PyramidSpatialAggregator<byte[], SparseIntFV>(
                    bovw, 2, 4);

            /** Spatial histograms are appended together and normalised */
            return spatial.aggregate(pdsift.getByteKeypoints(0.015f), image.getBounds()).normaliseFV();
        }
    }

}