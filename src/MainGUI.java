import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

import weka.classifiers.semi.MultiSemiAdaBoost;
import weka.classifiers.semi.SemiBoost;
import weka.gui.GenericObjectEditor;
import weka.gui.GenericPropertiesCreator;
import weka.gui.explorer.Explorer;

public class MainGUI {

	public static void main(String[] arg) throws IOException {
		List<Class<?>> loading = new ArrayList<>();
		loading.add(SemiBoost.class);
		loading.add(MultiSemiAdaBoost.class);

		Properties properties;
		String classifiers = "weka.classifiers.Classifier";
		for (Class<?> c : loading) {
			properties = GenericPropertiesCreator.getGlobalInputProperties();
			properties.setProperty(classifiers, properties.getProperty(classifiers) + "," + c.getPackage());

			properties = GenericPropertiesCreator.getGlobalOutputProperties();
			properties.setProperty(classifiers, properties.getProperty(classifiers) + "," + c.getCanonicalName());
		}
		GenericObjectEditor.determineClasses();

		Explorer.main(new String[] { "iris3.semi.arff" });
	}
}
