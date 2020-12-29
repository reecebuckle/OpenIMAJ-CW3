package uk.ac.soton.ecs.cw3;

import de.bwaldvogel.liblinear.SolverType;
import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.sampling.GroupSampler;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.feature.*;
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
 * You should develop a set of linear classifiers:
 * Use the LiblinearAnnotator class to automatically create 15 one-vs-all classifiers
 * Use a bag-of-visual-words feature based on fixed size densely-sampled pixel patches
 * <p>
 * We recommend that you start with 8x8 patches, sampled every 4 pixels in the x and y directions.
 * A sample of these should be clustered using K-Means to learn a vocabulary (try ~500 clusters to start).
 * You might want to consider mean-centring and normalising each patch before clustering/quantisation.
 * <p>
 * Note: we’re not asking you to use SIFT features here -
 * just take the pixels from the patches and flatten them into a vector &
 * then use vector quantisation to map each patch to a visual word.
 */
public class LinearClassifier {

    //Might not be necessary
    private LiblinearAnnotator<FImage, String> annotator;

    /**
     * @param args
     * @throws FileSystemException
     * @implements LibLinearAnnotator http://openimaj.org/apidocs/org/openimaj/ml/annotation/linear/LiblinearAnnotator.html
     */
    public static void main(String[] args) throws FileSystemException {

        //Load training dataset
        //TODO: Implement a better way of loading the dataset and individual images, just using this as a default
        GroupedDataset<String, VFSListDataset<FImage>, FImage> trainingDataset =
                new VFSGroupDataset<FImage>(System.getProperty("user.dir") + "/OpenIMAJ-CW3/training", ImageUtilities.FIMAGE_READER);

        //Instantiate patch extractor for densely-sampled pixel patches
        //TODO: Write up patch extractor method
        PatchExtractor patchExtractor = new PatchExtractor();
        //Instantiate hard assigner
        //TODO: Parse the trainingDataset of correct type
        HardAssigner<float[], float[], IntFloatPair> assigner = trainQuantiser(trainingDataset,patchExtractor);

        //Use Bag of Visual Words Extractor (no need for SIFT features as shown in Chapter 12)
        FeatureExtractor<DoubleFV, FImage> extractor = new BOVWExtractor(patchExtractor,assigner);

        //Use LiblinearAnnotator to construct and train classifier
        LiblinearAnnotator<FImage, String> annotator = new LiblinearAnnotator<>(
                extractor, LiblinearAnnotator.Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);

        //Train the dataset using the openimaj's annotator
        annotator.train(trainingDataset);

        ClassificationEvaluator<CMResult<String>, String, FImage> eval =
                new ClassificationEvaluator<CMResult<String>, String, FImage>(
                        annotator, trainingDataset, new CMAnalyser<FImage, String>(CMAnalyser.Strategy.SINGLE));

        Map<FImage, ClassificationResult<String>> guesses = eval.evaluate();
        CMResult<String> result = eval.analyse(guesses);

        System.out.println(result.getDetailReport());

    }

    /**
     * Base code adapted from HardAssigner class provided in Chapter 12
     * Not too sure about the exact methods and data types of this
     * <p>
     * TODO: To write hard assigner that can assign features (from patch extractor) to identifiers
     */
    public static HardAssigner<float[], float[], IntFloatPair> trainQuantiser(GroupedDataset<String, VFSListDataset<FImage>, FImage> trainingDataset, PatchExtractor extractor) {

        //Some sort of list here to hold in features, as per Chap 12?
        List<float[]> allkeys = new ArrayList<float[]>();

        //Iterate through training dataset (utilising Josh's GroupedDataset method)
        for (FImage image : trainingDataset) {
            List<LocalFeature<SpatialLocation, FloatFV>> sampleList = extractor.extract(image, 4, 8);

            for (LocalFeature<SpatialLocation, FloatFV> localFeature : sampleList) {
                allkeys.add(localFeature.getFeatureVector().values);
            }
        }

        //A sample of these should be clustered using K-Means to learn a vocabulary (try ~500 clusters to start).
        FloatKMeans km = FloatKMeans.createKDTreeEnsemble(500);
        float[][] datasource = allkeys.toArray(new float[][]{});

        FloatCentroidsResult result = km.cluster(datasource);
        return result.defaultHardAssigner();
    }

    public static class PatchExtractor {

        public LocalFeature<SpatialLocation, FloatFV> getLocalFeature(FImage image, Rectangle rectangle) {
            image.processInplace(new MeanCenter());
            image.normalise();
            FloatFV feature = new FloatFV(image.getFloatPixelVector());
            return new LocalFeatureImpl<SpatialLocation, FloatFV>(
                    new SpatialLocation(rectangle.x, rectangle.y), feature);
        }

        public List<LocalFeature<SpatialLocation, FloatFV>> extract(FImage image, int step, int size) {
            RectangleSampler sampler = new RectangleSampler(image, step, step, size, size);
            List<LocalFeature<SpatialLocation, FloatFV>> features = new ArrayList<LocalFeature<SpatialLocation, FloatFV>>();

            for (Rectangle rectange : sampler) {
                FImage patch = image.extractROI(rectange);
                LocalFeature<SpatialLocation, FloatFV> localFeature = getLocalFeature(patch, rectange);
                features.add(localFeature);
            }

            return features;
        }
    }

    /**
     * Implements FeatureExtractor class
     * Not too sure on this methods integration with "PatchExtractor", I know somewhere we need to
     * Sample 8x8 patches every 4 pixels (in X and Y direction)
     * <p>
     * Adapted from chapter 12's PHOWExtractor Implementation:
     * http://openimaj.org/tutorial/classification101.html
     *
     * @implements FeatureExtractor interface http://openimaj.org/apidocs/org/openimaj/feature/FeatureExtractor.html
     * @implements BagOfVisualWords: http://openimaj.org/apidocs/org/openimaj/image/feature/local/aggregate/BagOfVisualWords.html
     */
    public static class BOVWExtractor implements FeatureExtractor<DoubleFV, FImage> {
        PatchExtractor patchExtractor;
        HardAssigner<float[], float[], IntFloatPair> assigner;

        // Constructor
        public BOVWExtractor(PatchExtractor patchExtractor, HardAssigner<float[], float[], IntFloatPair> assigner) {
            this.patchExtractor = patchExtractor;
            this.assigner = assigner;
        }

        // Implemented Methods
        public DoubleFV extractFeature(FImage image) {
            //Extracts basic (hard-assignment) bag of visual words (BOVW) representations of an image given a list of
            //local features and an HardAssigner
            BagOfVisualWords<float[]> bovw = new BagOfVisualWords<float[]>(assigner);

            //Use BlockSpatialAggregator together with BagOfVisualWords
            BlockSpatialAggregator<float[], SparseIntFV> spatial =
                    new BlockSpatialAggregator<float[], SparseIntFV>(bovw, 2, 2);

            //public SparseIntFV aggregate(List<? extends LocalFeature<?,? extends ArrayFeatureVector<T>>> features)
            return spatial.aggregate(patchExtractor.extract(image, 4, 8), image.getBounds()).normaliseFV();
        }
    }
}



