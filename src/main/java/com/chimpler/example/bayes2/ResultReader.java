package com.chimpler.example.bayes2;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.HashMap;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.classifier.naivebayes.BayesUtils;

public class ResultReader {
	
	public static Map<String, Integer> readCategoryByTweetIds(Configuration configuration, String tweetFileName) throws Exception {
		Map<String, Integer> categoryByTweetIds = new HashMap<String, Integer>();
		BufferedReader reader = new BufferedReader(new FileReader(tweetFileName));
		while(true) {
			String line = reader.readLine();
			if (line == null) {
				break;
			}
			String[] tokens = line.split("\t", 2);
			String tweetId = tokens[0];
			Integer categoryId = Integer.parseInt(tokens[1]);
			categoryByTweetIds.put(tweetId, categoryId);
		}
		reader.close();
		return categoryByTweetIds;
	}
	
	public static void main(String[] args) throws Exception {
		if (args.length < 3) {
			System.out.println("Arguments: [tweet file] [label index] [tweet category ids]");
			return;
		}
		
		String tweetFileName = args[0];
		String labelIndexPath = args[1];
		String tweetCategoryIdsPath = args[2];
		
		Configuration configuration = new Configuration();

		Map<String, Integer> categoryByTweetIds = readCategoryByTweetIds(configuration, tweetCategoryIdsPath);
		Map<Integer, String> labels = BayesUtils.readLabelIndex(configuration, new Path(labelIndexPath));

		BufferedReader reader = new BufferedReader(new FileReader(tweetFileName));
		while(true) {
			String line = reader.readLine();
			if (line == null) {
				break;
			}
			String[] tokens = line.split("\t", 2);
			String tweetId = tokens[0];
			String tweet = tokens[1];
			int categoryId = categoryByTweetIds.get(tweetId);
			System.out.println(tweetId + ": " + tweet);
			System.out.println(" => " + labels.get(categoryId));
		}
		reader.close();

	}
}
