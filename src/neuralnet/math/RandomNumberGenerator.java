package neuralnet.math;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.Random;

/**
 * RandomNumberGenerator
 * This class generates double precision random numbers according to a seed. It
 * is used in weights initialization, for example.
 *
 * @author Alan de Souza, Fábio Soares
 * @version 0.1
 */
public class RandomNumberGenerator {
    /**
     * Seed that is used for random number generation
     */
    public static long seed = 0;
    /**
     * Random singleton object that actually generates the random numbers
     */
    public static Random r;

    /**
     * GenerateNext
     * Static method that returns a newly random number
     *
     * @return
     */
    public static double GenerateNext() {
        if (r == null)
            r = new Random(seed);
        return r.nextDouble();
    }

    /**
     * setSeed
     * Sets a new seed for the random generator
     *
     * @param seed new seed for random generator
     */
    public static void setSeed(long seed) {
        RandomNumberGenerator.seed = seed;
        r.setSeed(seed);
    }

    public static int[] hashInt(int start, int end) {
        LinkedList<Integer> ll = new LinkedList<>();
        ArrayList<Integer> al = new ArrayList<>();
        for (int i = start; i <= end; i++) {
            ll.add(i);
        }
        int start0 = 0;
        for (int end0 = end - start; end0 > start0; end0--) {
            int rnd = RandomNumberGenerator.GenerateIntBetween(start0, end0);
            int value = ll.get(rnd);
            ll.remove(rnd);
            al.add(value);
        }
        al.add(ll.get(0));
        ll.remove(0);
        return ArrayOperations.arrayListToIntVector(al);
    }

    public static double GenerateBetween(double min, double max) {
        if (r == null)
            r = new Random(seed);
        if (max < min)
            return min;
        return min + (r.nextDouble() * (max - min));
    }

    public static int GenerateIntBetween(int min, int max) {
        if (r == null)
            r = new Random(seed);
        if (max < min)
            return min;
        return min + (r.nextInt(max - min));
    }

}
