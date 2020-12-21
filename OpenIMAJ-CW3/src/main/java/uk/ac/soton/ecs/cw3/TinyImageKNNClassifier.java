package uk.ac.soton.ecs.cw3;

import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
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
import org.openimaj.ml.annotation.basic.KNNAnnotator;
import org.openimaj.util.array.ArrayUtils;


import java.io.IOException;
import java.util.Map;

public class TinyImageKNNClassifier {

    /*
    You should develop a simple k-nearest-neighbour classifier
    using the “tiny image” feature. The “tiny image” feature is one of the
    simplest possible image representations. One simply crops each image to
    a square about the centre, and then resizes it to a small, fixed resolution
    (we recommend 16x16). The pixel values can be packed into a vector by
    concatenating each image row. It tends to work slightly better if the tiny
    image is made to have zero mean and unit length.
     */

    static class TinyExtractor implements FeatureExtractor<DoubleFV, FImage> {
        public DoubleFV extractFeature(FImage image) {

            // Size of the tiny image square
            final int tinySize = 16;

            // Compute the maximum size the n*n cropped image can be
            int croppedSize = Math.min(image.width,image.height);

            // Crop the image to the maximum dimensions
            FImage cropped = image.extractCenter(croppedSize,croppedSize);

            // Resize image to a 16 x 16 version
            FImage tiny = cropped.process(new ResizeProcessor(tinySize, tinySize));

            return new DoubleFV(ArrayUtils.reshape(ArrayUtils.convertToDouble(tiny.pixels)));
        }
    }

    public static void main(String[] args) throws IOException {

        // Load the dataset
        GroupedDataset<String, VFSListDataset<FImage>, FImage> images =
                new VFSGroupDataset<FImage>(System.getProperty("user.dir")+"/OpenIMAJ-CW3/training", ImageUtilities.FIMAGE_READER);

        TinyExtractor extractor = new TinyExtractor();

        KNNAnnotator<FImage, String, DoubleFV> ann = KNNAnnotator.create(extractor, DoubleFVComparison.EUCLIDEAN, 4);


        GroupedRandomSplitter<String, FImage> splits =
                new GroupedRandomSplitter<String, FImage>(images, 15, 0, 15);

        ann.train(splits.getTrainingDataset());

        ClassificationEvaluator<CMResult<String>, String, FImage> eval =
                new ClassificationEvaluator<CMResult<String>, String, FImage>(
                        ann, splits.getTestDataset(), new CMAnalyser<FImage, String>(CMAnalyser.Strategy.SINGLE));

        Map<FImage, ClassificationResult<String>> guesses = eval.evaluate();
        CMResult<String> result = eval.analyse(guesses);

        System.out.println(result.getDetailReport());

    }
}
