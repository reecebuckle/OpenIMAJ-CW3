package uk.ac.soton.ecs.cw3;

import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.colour.ColourSpace;
import org.openimaj.image.colour.RGBColour;
import org.openimaj.image.processing.convolution.FGaussianConvolve;
import org.openimaj.image.typography.hershey.HersheyFont;

import java.io.IOException;

/**
 * Hi team, hope you guys all got here!
 *
 */
public class App {
    public static void main( String[] args ) {

        try {
            CNN cnn = new CNN();
        } catch (InterruptedException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
