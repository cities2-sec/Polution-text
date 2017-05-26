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

import weka.classifiers.trees.J48;
import weka.classifiers.lazy.IBk;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Classifier;

import java.io.FileNotFoundException;

/**
 * This example class trains a J48, IBK and NaiveBayes classifier on a dataset
 *
 * @author  FracPete (fracpete at waikato dot ac dot nz)
 * @version $Revision: 5496 $
 */
public class CreateModels {

  /**
   * Expects two parameters: training file and test files
   * @param args	the commandline arguments
   * @throws Exception	if something goes wrong
   */
  public static void main(String[] args) throws Exception {
    // load data
    // load train data
    DataSource source = new DataSource("./data/diabetes.arff");
    Instances train = source.getDataSet();
    train.setClassIndex(train.numAttributes() - 1);

    // train classifier
/*
    Classifier [] classifiers = { new J48(), new NaiveBayes(), new IBk()};
    for (Classifier c : classifiers){
      Classifier cls = c;
      cls.buildClassifier(train);
*/
    J48 cls1 = new J48();
    cls1.buildClassifier(train);
    IBk cls2 = new IBk();
    cls2.buildClassifier(train);
    NaiveBayes cls3 = new NaiveBayes();
    cls3.buildClassifier(train);

    //serialize model
    try {
      weka.core.SerializationHelper.write("./models/j48.model", cls1);
      weka.core.SerializationHelper.write("./models/IBk.model", cls2);
      weka.core.SerializationHelper.write("./models/NaiveBayes.model", cls3);
      System.out.print("Models created!");
    }
    catch (FileNotFoundException e){
      System.out.print(e);
    }

  }
}
