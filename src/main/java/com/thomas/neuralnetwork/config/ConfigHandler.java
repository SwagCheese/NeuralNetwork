package com.thomas.neuralnetwork.config;

public class ConfigHandler {
    private ConfigHandler instance;


    private ConfigHandler() {

    }

    public ConfigHandler getInstance() {
        if (instance == null) instance = new ConfigHandler();

        return instance;
    }

    public boolean load() {

        return true;
    }
}
