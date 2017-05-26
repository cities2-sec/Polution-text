/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 * CreateModels.java
 * Copyright (C) 2009 University of Waikato, Hamilton, New Zealand
 */

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import sun.tools.jar.Main;
import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.core.Instance;
import weka.core.DenseInstance;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;

import java.io.FileNotFoundException;
import java.util.Scanner;

/**
 * This example class trains a J48 classifier on a dataset and outputs for
 * a second dataset the actual and predicted class label, as well as the
 * class distribution.
 *
 * @author FracPete (fracpete at waikato dot ac dot nz)
 * @version $Revision: 5496 $
 */
public class EvaluateModels {
    /**
     * Expects two parameters: training file and test file.
     *
     * @param args the commandline arguments
     * @throws Exception if something goes wrong
     */
    @Parameter(names={"--day", "-d"})
    static
    String day;
    @Parameter(names={"--no2", "-n"})
    static
    double no2;
    @Parameter(names={"--pm10", "-p"})
    static
    double pm10;
    @Parameter(names={"--temperature", "-t"})
    static
    double temperature;
    @Parameter(names={"--location", "-l"})
    static
    String location;
    @Parameter(names={"--model", "-m"})
    static
    String model;
    @Parameter(names={"--datafile", "-f"})
    static
    String data;


    public static void main(String[] args) throws Exception {

        EvaluateModels main = new EvaluateModels();
        JCommander.newBuilder()
                .addObject(main)
                .build()
                .parse(args);

        Instances polutioninstance = NewInstance(day, no2, pm10, temperature, location, data);
        try {
            String[] filePath = {model};
            // load test data
            //DataSource source_test = new DataSource("./data/polution.arff");
            Instances test = polutioninstance;
            //System.out.println(test);

            test.setClassIndex(test.numAttributes() - 1);

            for (String path : filePath) {
                // Deserialize model
                Classifier cls = (Classifier) weka.core.SerializationHelper.read(path);
                //Evaluate with croos-validation

                // output predictions
                System.out.println("*****************************");
                System.out.println("This is the Model: " + path.toString());
                System.out.println("*****************************");
                System.out.println("predicted  - distribution");
                double pred = cls.classifyInstance(test.instance(0));
                double[] dist = cls.distributionForInstance(test.instance(0));
                System.out.print(test.classAttribute().value((int) pred));
                System.out.print(" - ");
                System.out.print(Utils.arrayToString(dist));
            }
        } catch (FileNotFoundException e) {
            System.out.print(e);
        }
    }

    public static Instances NewInstance(String day, double no2, double pm10, double tmp, String place, String filedata) throws Exception {
        // load dataset
        Instances data = DataSource.read(filedata);
        // Constructor creating an empty set of instances. Copies references to the header information from the given set of instances.
        Instances newData = null;
        newData = new Instances(data, 1);

        // Create empty instance with 5 attribute values
        Instance inst = new DenseInstance(7);

        // Set instance's dataset to be the dataset "race"
        inst.setDataset(data);

        //Scanner reader = new Scanner(System.in);
        //System.out.println("Day of the week: ");
        //String day = reader.next();
        inst.setValue(0, day);
        //System.out.println("NO2: ");
        //double no2 = reader.nextDouble();
        inst.setValue(1, no2);
        //System.out.println("PM10: ");
        //double pm10 = reader.nextDouble();
        inst.setValue(2, pm10);
        //System.out.println("Temperature");
        //double tmp = reader.nextDouble();
        inst.setValue(3, tmp);
        //System.out.println("Polution: ?");
        //String polution = reader.next();
        //inst.setValue(4, polution);
        inst.setMissing(4);
        //System.out.println("Place {Palau_Reial, Torrente_Gornal, Sants}: ");
        //String place = reader.next();
        inst.setValue(5, place);

        // Print the instance
        //System.out.println("The instance: " + inst);

        newData.add(inst);
        //System.out.println(data);
        //System.out.println(newData);

        return newData;
    }
}
