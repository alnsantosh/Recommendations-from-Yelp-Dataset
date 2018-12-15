import com.mongodb.spark.MongoSpark;
import com.mongodb.spark.config.ReadConfig;
import com.mongodb.spark.rdd.api.java.JavaMongoRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.ForeachFunction;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.bson.Document;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

public class UserRestaurantFilter {
    public static void main(String[] args) {
        SparkSession sparkSession=SparkSession.builder()
                .master("local[*]")
                .appName("MongoSparkConnector")
                .config("spark.mongodb.input.uri", "mongodb://127.0.0.1/yelp.business")
                //.config("spark.mongodb.output.uri", "mongodb://student:student@54.173.174.196/test.myCollection")
                .getOrCreate();
        JavaSparkContext jsc=new JavaSparkContext(sparkSession.sparkContext());
        //Creating a ReadConfig in order to read another collection from mongo db
        Map<String,String> readOverride=new HashMap<String, String>();
        readOverride.put("collection","review");
        readOverride.put("readPreference.name","secondaryPreferred");
        ReadConfig readConfig=ReadConfig.create(jsc).withOptions(readOverride);

        JavaMongoRDD<Document> reviewRDD=MongoSpark.load(jsc,readConfig);
        JavaMongoRDD<Document> businessRDD=MongoSpark.load(jsc);
        Dataset<Row> review=reviewRDD.toDF();
        Dataset<Row> business=businessRDD.toDF();
        review.createOrReplaceTempView("review");
        business.createOrReplaceTempView("business");
        Dataset<Row> joined=sparkSession.sql("select r.user_id,b.city,b.latitude,b.longitude from business b,review r where b.business_id=r.business_id");
        joined.foreach((ForeachFunction<Row>) UserRestaurantFilter::call);
        joined.printSchema();

    }
    private static void call(Row row) throws IOException {
        BufferedWriter writer = new BufferedWriter(new FileWriter("F:\\output_bucket1.txt",true));
        writer.append(row.getAs("user_id")+"\t"+row.getAs("city")+ "\t"+row.getAs("latitude")+ "\t"+row.getAs("longitude")+"\n");
        writer.close();
    }
}
