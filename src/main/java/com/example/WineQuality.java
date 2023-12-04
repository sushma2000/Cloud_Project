package com.example;

import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.feature.StandardScaler;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.param.ParamMap;

import java.io.IOException;
import java.util.Arrays;

public class WineQuality {
    public static void main(String[] args) throws IOException {
        SparkSession spark = SparkSession.builder().appName("Wine Quality").getOrCreate();

        // Load and process data
        Dataset<Row> trainData = loadAndProcessData(spark, "s3://wine-quality-prediction/TrainingDataset.csv");
        Dataset<Row> validationData = loadAndProcessData(spark, "s3://wine-quality-prediction/ValidationDataset.csv");

        // Logistic Regression Model
        LogisticRegression lr = new LogisticRegression().setLabelCol("quality").setFeaturesCol("scaledFeatures")
                .setMaxIter(10).setRegParam(0.3);
        LogisticRegressionModel lrModel = lr.fit(trainData);

        // Evaluate Logistic Regression Model
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator().setLabelCol("quality")
                .setPredictionCol("prediction");
        double trainAccuracy = evaluator.evaluate(lrModel.transform(trainData));
        double validationAccuracy = evaluator.evaluate(lrModel.transform(validationData)
        );

        System.out.println("Train Accuracy: " + trainAccuracy + "\nValidation Accuracy: " + validationAccuracy);

        // Random Forest Model with Cross Validation
        RandomForestClassifier rf = new RandomForestClassifier().setLabelCol("quality").setFeaturesCol("scaledFeatures");

        ParamGridBuilder paramGridBuilder = new ParamGridBuilder();
        ParamMap[] paramGrid = paramGridBuilder.addGrid(rf.numTrees(), new int[] { 20, 50, 100 })
                .addGrid(rf.maxDepth(), new int[] { 5, 10, 15 }).build();

        CrossValidator cv = new CrossValidator().setEstimator(rf).setEvaluator(evaluator).setEstimatorParamMaps(paramGrid)
                .setNumFolds(3);

        CrossValidatorModel cvModel = cv.fit(trainData);
        double bestScore = cvModel.avgMetrics()[0];

        System.out.println("Best Validation F1 Score: " + bestScore);

        String bestModelPath = "s3://wine-quality-prediction/best_model";
        RandomForestClassificationModel bestRfModel = (RandomForestClassificationModel) cvModel.bestModel();
        bestRfModel.write().overwrite().save(bestModelPath);
        System.out.println("Best model saved to " + bestModelPath);

        spark.stop();
    }

    private static Dataset<Row> loadAndProcessData(SparkSession spark, String filePath) {
        Dataset<Row> df = spark.read().option("header", "true").option("sep", ";").option("inferSchema", "true")
                .csv(filePath);

        String[] columns = new String[] { "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
                "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol",
                "quality" };
        df = df.toDF(columns);

        VectorAssembler assembler = new VectorAssembler().setInputCols(Arrays.copyOfRange(columns, 0, columns.length - 1))
                .setOutputCol("features");
        StandardScaler scaler = new StandardScaler().setInputCol("features").setOutputCol("scaledFeatures")
                .setWithStd(true).setWithMean(true);

        Pipeline pipeline = new Pipeline().setStages(new PipelineStage[] { assembler, scaler });
        PipelineModel pipelineModel = pipeline.fit(df);
        return pipelineModel.transform(df);
    }
}
