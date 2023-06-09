module com.thomas.neuralnetwork {
    requires javafx.controls;
    requires javafx.fxml;

    requires org.controlsfx.controls;

    opens com.thomas.neuralnetwork to javafx.fxml;
    exports com.thomas.neuralnetwork;
    exports com.thomas.neuralnetwork.controllers;
    opens com.thomas.neuralnetwork.controllers to javafx.fxml;
}