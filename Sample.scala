import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.UserDefinedFunction

object AddressMatching {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("Address Matching")
      .master("local[*]")
      .getOrCreate()

    val inputPath = "/edl/hdfs/jffv-mns/testaddress/input/query-impala-88364.csv"
    val mainOutputPath = "/edl/hdfs/jffv-mns/testaddress/output/main_output.csv"
    val duplicateOutputPath = "/edl/hdfs/jffv-mns/testaddress/output/duplicates.csv"

    // Read input data
    val data = spark.read.option("header", "true").csv(inputPath)

    // Normalize address and postal code
    val normalizeUDf: UserDefinedFunction = udf((address: String) => {
      Option(address).map(_.toLowerCase
        .replaceAll("\\b(st|road|rd|avenue|ave|boulevard|blvd|way|circle|pk|park)\\b", "st")
        .replaceAll("[^a-z0-9\\s]", "")
        .trim).orNull
    })
    val normalizedDf = data
      .withColumn("normalized_address", normalizeUDf(col("address")))
      .withColumn("normalized_postal_cd", col("postal_cd").substr(1, 5))

    // Tokenization and StopWords removal
    val tokenizer = new org.apache.spark.ml.feature.Tokenizer()
      .setInputCol("normalized_address")
      .setOutputCol("tokens")
    val tokenizedDf = tokenizer.transform(normalizedDf)

    val remover = new org.apache.spark.ml.feature.StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("filtered_tokens")
    val filteredDf = remover.transform(tokenizedDf)

    // Group addresses by postal code and compare
    val groupedDf = filteredDf.groupBy("normalized_postal_cd")
      .agg(collect_list(struct("aiops_site_id", "normalized_address", "filtered_tokens")).as("address_group"))

    val similarityUDF = udf((addresses: Seq[Row]) => {
      val results = scala.collection.mutable.ArrayBuffer[(String, String, Double)]()
      for (i <- addresses.indices; j <- i + 1 until addresses.size) {
        val addr1 = addresses(i)
        val addr2 = addresses(j)
        val tokens1 = addr1.getAs[Seq[String]]("filtered_tokens")
        val tokens2 = addr2.getAs[Seq[String]]("filtered_tokens")
        val id1 = addr1.getAs[String]("aiops_site_id")
        val id2 = addr2.getAs[String]("aiops_site_id")
        val intersection = tokens1.intersect(tokens2).size.toDouble
        val union = tokens1.union(tokens2).distinct.size.toDouble
        val score = if (union > 0) intersection / union else 0.0
        results += ((id1, id2, score))
      }
      results
    })

    val comparisonDf = groupedDf.withColumn("comparisons", similarityUDF(col("address_group")))
    val flatDf = comparisonDf.select(explode(col("comparisons")).as("comparison"))
      .select(
        col("comparison._1").as("aiops_site_id_1"),
        col("comparison._2").as("aiops_site_id_2"),
        col("comparison._3").as("similarity_score")
      )
    val duplicateDf = flatDf.filter(col("similarity_score") === 1.0)

    // Get non-duplicate records
    val duplicateIds = duplicateDf
      .select(col("aiops_site_id_1"))
      .union(duplicateDf.select(col("aiops_site_id_2")))
      .distinct()

    val nonDuplicateDf = filteredDf.join(duplicateIds, filteredDf("aiops_site_id") === duplicateIds("aiops_site_id_1"), "left_anti")

    // Enrich duplicates with address details
    val enrichedDuplicatesDf = duplicateDf
      .join(filteredDf.select(
        col("aiops_site_id").as("aiops_site_id_1"),
        col("address").as("address_1"),
        col("normalized_address").as("normalized_address_1")
      ), Seq("aiops_site_id_1"))
      .join(filteredDf.select(
        col("aiops_site_id").as("aiops_site_id_2"),
        col("address").as("address_2"),
        col("normalized_address").as("normalized_address_2")
      ), Seq("aiops_site_id_2"))

    // Flatten and write non-duplicate records
    val flatNonDuplicateDf = nonDuplicateDf
      .withColumn("tokens_str", concat_ws(" ", col("tokens")))
      .withColumn("filtered_tokens_str", concat_ws(" ", col("filtered_tokens")))
      .drop("tokens", "filtered_tokens")

    flatNonDuplicateDf.write
      .mode("overwrite")
      .option("header", "true")
      .csv(mainOutputPath)

    // Write enriched duplicate records
    enrichedDuplicatesDf.write
      .mode("overwrite")
      .option("header", "true")
      .csv(duplicateOutputPath)

    println(s"Main results have been written to $mainOutputPath")
    println(s"Duplicate records have been written to $duplicateOutputPath")
  }
}
