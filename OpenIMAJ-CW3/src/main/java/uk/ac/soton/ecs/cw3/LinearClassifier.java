package uk.ac.soton.ecs.cw3;

import de.bwaldvogel.liblinear.SolverType;
import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.util.pair.IntFloatPair;

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

    /**
     * @param args
     * @throws FileSystemException
     */
    public static void main(String[] args) throws FileSystemException {

        //Load training dataset
        GroupedDataset<String, VFSListDataset<FImage>, FImage> images =
                new VFSGroupDataset<FImage>(System.getProperty("user.dir") + "/OpenIMAJ-CW3/training", ImageUtilities.FIMAGE_READER);

        //Instantiate hard assigner
        CustomHardAssigner assigner = new CustomHardAssigner();

        //Instantiate patch extractor for densely-sampled pixel patches
        PatchExtractor patchExtractor = new PatchExtractor();

        //Use Bag of Visual Words Extractor (no need for SIFT features as shown in Chapter 12)
        FeatureExtractor<SparseIntFV, FImage> extractor = new BOVWExtractor(patchExtractor, assigner);

        //Use LiblinearAnnotator to construct and train classifier
        LiblinearAnnotator<FImage, String> annator = new LiblinearAnnotator<>(
                extractor, LiblinearAnnotator.Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);

        //Split the loaded dataset
        //GroupedRandomSplitter<String, FImage> splits = new GroupedRandomSplitter<String, FImage>(images, 15, 0, 15);

        //Train dataset
        //ann.train(splits.getTrainingDataset());

    }

    //adapted form http://openimaj.org/apidocs/org/openimaj/feature/BagOfWordsFeatureExtractor.html


    /**
     * Implements FeatureExtractor class and implemented methods
     * TODO: finish integration of patch extractor and hard assigner
     */
    public static class BOVWExtractor implements FeatureExtractor<SparseIntFV, FImage> {

        //PatchExtractor patchExtractor;
        PatchExtractor patchExtractor;
        CustomHardAssigner assigner;

        // Constructor
        public BOVWExtractor(PatchExtractor patchExtractor, CustomHardAssigner assigner) {
            this.patchExtractor = patchExtractor;
            this.assigner = assigner;
        }

        // Implemented Methods
        @Override
        public SparseIntFV extractFeature(FImage object) {
            return null;
        }
    }

    /**
     * TODO: to write something that can extract 8x8 patches every 4 pixels etc...
     */
    public static class PatchExtractor {


    }

    /**
     * Adapted from HardAssigner class provided in Chapter 12
     * TODO: To write hard assigner that can assign features (from patch extractor) to identifiers
     */
    private static class CustomHardAssigner {

    }


}
