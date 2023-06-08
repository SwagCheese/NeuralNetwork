package com.thomas.neuralnetwork.data;

import java.io.*;

public class UserData {
    private static final UserData instance = new UserData();

    private UserData() {}

    public static UserData getInstance() {
        return instance;
    }

    public byte[] convertPixelsToByteArray(int[][] pixels) {
        byte[] data = new byte[784]; // 28x28 = 784
        int index = 0;

        for (int y = 0; y < 28; y++) {
            for (int x = 0; x < 28; x++) {
                data[index++] = (byte) pixels[x][y];
            }
        }

        return data;
    }

    public void addToFile(String filename, byte[] data) {
        incrementFirstDimensionInFile(filename);
        try (FileOutputStream fos = new FileOutputStream(filename, true);
             BufferedOutputStream bos = new BufferedOutputStream(fos)) {
            bos.write(data);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public boolean fileNotFound(String filename) {
        File file = new File(filename);
        return !file.exists() || file.isDirectory();
    }

    public void setupImageFile(String imagesFilename) {
        try (FileOutputStream fos = new FileOutputStream(imagesFilename);
             BufferedOutputStream bos = new BufferedOutputStream(fos)) {
            /*
            Write the magic number
            The magic number is four bytes long. The first 2 bytes are always 0.

            The third byte codes the type of the data:
                0x08: unsigned byte
                0x09: signed byte
                0x0B: short (2 bytes)
                0x0C: int (4 bytes)
                0x0D: float (4 bytes)
                0x0E: double (8 bytes)

            The fourth byte codes the number of dimensions in the vector/matrix
             */
            bos.write(new byte[]{0x00, 0x00, 0x08, 0x03});

            // Write the dimensions (0 images, 28x28 pixels)
            // Dimension 1 will be updated later
            bos.write(0); // Size in dimension 1 (0)
            bos.write(new byte[]{0x00, 0x00, 0x00, 0x1C}); // Size in dimension 2 (28)
            bos.write(new byte[]{0x00, 0x00, 0x00, 0x1C}); // Size in dimension 3 (28)
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void setupLabelFile(String labelsFilename) {
        try (FileOutputStream fos = new FileOutputStream(labelsFilename);
             BufferedOutputStream bos = new BufferedOutputStream(fos)) {
            /*
            Write the magic number
            The magic number is four bytes long. The first 2 bytes are always 0.

            The third byte codes the type of the data:
                0x08: unsigned byte
                0x09: signed byte
                0x0B: short (2 bytes)
                0x0C: int (4 bytes)
                0x0D: float (4 bytes)
                0x0E: double (8 bytes)

            The fourth byte codes the number of dimensions in the vector/matrix
            */
            bos.write(new byte[]{0x00, 0x00, 0x08, 0x01});

            // Write the dimensions
            bos.write(0); // Number of dimensions
            bos.write(new byte[]{0x00, 0x00, 0x00, 0x00}); // Size in dimension 1 (0)
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void incrementFirstDimensionInFile(String filename) {
        try (RandomAccessFile file = new RandomAccessFile(filename, "rw")) {
            file.seek(4);
            file.writeInt(file.readInt()+1);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
