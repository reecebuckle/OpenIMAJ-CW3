package uk.ac.soton.ecs.cw3;

import de.bwaldvogel.liblinear.SolverType;
import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
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
import org.openimaj.ml.annotation.ScoredAnnotation;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;
import org.openimaj.util.pair.IntFloatPair;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

public class LinearClassifier extends Classifier {
    /**
     * LinearClassifier takes patches of pixels from images and performs vector quantisation to map each patch
     * to a visual word. A sample of these are clustered using K-Means to learn a vocabulary.
     */
    private final int STEP; // Pixels between patches
    private final int SIZE; // Size of patches
    private final int CLUSTERS; // Number of clusters in KNN

    LiblinearAnnotator<FImage, String> ann; // The annotator used for training the model

    /**
     * Constructor used to set parameters of patch extractor and number of clusters of KNN.
     *
     * @param step Number of steps between each patch in an image
     * @param size Size of the patches
     * @param clusters Number of clusters used in KNN
     */
    public LinearClassifier(int step, int size, int clusters) {
        this.STEP = step;
        this.SIZE = size;
        this.CLUSTERS = clusters;
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
        File test = new File(CWD+"\\OpenIMAJ-CW3\\testing");
        String[] filenames = test.list();
        sortFilenames(filenames);

        // Loops through files, then classifies them using the annotator and writes the results to the file.
        if (filenames != null) {
            for (String file : filenames) {
                FImage image = ImageUtilities.readF(new File(".\\OpenIMAJ-CW3\\testing\\"+file));
                List<ScoredAnnotation<String>> result = ann.annotate(image);
                fileWriter.write(String.format("%s %s\n",file, getClassification(result)));
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

        // Instantiate patch extractor for densely-sampled pixel patches
        // The feature extractor used to preprocess each image.
        PatchExtractor patchExtractor = new PatchExtractor();

        // Instantiate hard assigner
        HardAssigner<float[], float[], IntFloatPair> assigner = trainQuantiser(splits.getTrainingDataset(),patchExtractor);

        // Use Bag of Visual Words Extractor
        FeatureExtractor<DoubleFV, FImage> extractor = new BOVWExtractor(patchExtractor,assigner);

        // Use LiblinearAnnotator to construct and train classifier
        this.ann = new LiblinearAnnotator<>(
                extractor, LiblinearAnnotator.Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);

        // Trains the model.
        this.ann.train(splits.getTrainingDataset());
    }

    /**
     * Method used for testing the accuracy of the classifier
     *
     * @return Accuracy of the model
     */
    protected Double testClassifier() {
        Double n_correct = 0.0;
        int n = splits.getTestDataset().size();

        // Iterate over test set and classify each image
        // Compare against actual label
        for(Map.Entry<String, ListDataset<FImage>> set : splits.getTestDataset().entrySet()) {
            for(FImage image: set.getValue()) {
                String predicted = ann.classify(image).getPredictedClasses().toString().replaceAll("\\[|\\]", "");
                if (predicted.equals(set.getKey())) n_correct += 1;
            }
        }

        return n_correct / n;
    }

    /**
     * Method that quantises images using fixed size densely-sampled pixel patches
     * The vectors are then clustered using K-Means to learn a vocabulary
     *
     * @param trainingDataset The dataset to be quantised
     * @param extractor The extractor used to extract patches
     * @return HardAssigner that assigns the features to identifiers
     */
    public HardAssigner<float[], float[], IntFloatPair> trainQuantiser(GroupedDataset<String, ListDataset<FImage>,FImage> trainingDataset, PatchExtractor extractor) {
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

        // Cluster to learn vocabulary
        FloatKMeans km = FloatKMeans.createKDTreeEnsemble(CLUSTERS);
        float[][] datasource = allkeys.toArray(new float[][]{});

        FloatCentroidsResult result = km.cluster(datasource);

        return result.defaultHardAssigner();
    }

    public class PatchExtractor {

        /**
         * Method used to get features extracted from patches of an image
         *
         * @param image Image to extract the patches from
         * @param step Distance between patches
         * @param size Size of the patches
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
         * @param assigner Assigns features to identifiers
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
