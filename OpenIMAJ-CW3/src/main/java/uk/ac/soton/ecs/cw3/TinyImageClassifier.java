package uk.ac.soton.ecs.cw3;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.feature.FloatFVComparison;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.processing.algorithm.MeanCenter;
import org.openimaj.image.processing.resize.ResizeProcessor;
import org.openimaj.knn.FloatNearestNeighbours;
import org.openimaj.ml.clustering.assignment.soft.FloatKNNAssigner;

import java.util.Map.Entry;

public class TinyImageClassifier {

    VFSGroupDataset<FImage> data;
    VFSListDataset<FImage> testing;
    ConsoleHelper consoleHelper = new ConsoleHelper();

    public TinyImageClassifier() throws FileSystemException {

        getImages();
        preProcessImages(data, 16);

        GroupedRandomSplitter<String, FImage> splitter = new GroupedRandomSplitter<>(data, 70, 0, 30);
        System.out.println("Data Split... \nSplitting data");
        GroupedDataset<String, ListDataset<FImage>, FImage> train = splitter.getTrainingDataset();
        GroupedDataset<String, ListDataset<FImage>, FImage> test = splitter.getTestDataset();




        for (String key : train.getGroups()) System.out.println(key);


    }

    private void getImages() throws FileSystemException {

        String trainPath = "C:\\Users\\maxmr\\Documents\\Year 3\\Computer Vision\\Handin\\OpenIMAJ-CW3\\OpenIMAJ-CW3\\training";
        String testPath = "C:\\Users\\maxmr\\Documents\\Year 3\\Computer Vision\\Handin\\OpenIMAJ-CW3\\OpenIMAJ-CW3\\testing";

        data = new VFSGroupDataset<>(trainPath, ImageUtilities.FIMAGE_READER);
        testing = new VFSListDataset<>(testPath, ImageUtilities.FIMAGE_READER);

    }

    private void preProcessImages(VFSListDataset<FImage> data, String key, int size) {

        int i = 1;
        for (FImage image : data) {

            consoleHelper.animate(String.format("Pre Processing Images ... Class: %s -> %d.img", key, i));

            int width = image.getWidth();
            int height = image.getHeight();

            if (width != height) {
                int min = Math.min(width, height);
                image = image.extractCenter(min, min);
            }

            image.processInplace(new ResizeProcessor(size, size));
            image.processInplace(new MeanCenter());
            image.normalise();

            i++;

        }
    }

    private void preProcessImages(VFSGroupDataset<FImage> data, int size) {

        for (Entry<String,VFSListDataset<FImage>> entry : data.entrySet()) {
            preProcessImages(entry.getValue(), entry.getKey(), size);
        }
    }


}
