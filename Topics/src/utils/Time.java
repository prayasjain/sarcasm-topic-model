package utils;

public class Time {
	public static String current() {
		return (new java.text.SimpleDateFormat("yyyy.MM.dd HH:mm:ss")).format(new java.util.Date());
	}
}
