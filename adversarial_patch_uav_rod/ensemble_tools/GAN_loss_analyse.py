# coding = UTF-8
import os
import re
import matplotlib.pyplot as plt
import scipy
import numpy as np
root_file_path = './TrainSimpleGan3'
logger_file_path = os.path.join(root_file_path, 'logger.txt')
logger_save_path = os.path.join(root_file_path, 'log_analyse.png')


iterations = []
loss_D = []
loss_G = []
loss_D_real = []
loss_D_fake = []

pattern = r'\[(\d+)/(\d+)\]\[(\d+)/(\d+)\]'
with open(logger_file_path, 'r') as file:
    for line in file:
        if line.startswith('['):
            match = re.match(pattern, line)
            epoch, total_epoch, iteration, total_iteration = int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))
            parts = line.strip().split()

            loss_D.append(float(parts[2]))
            loss_G.append(float(parts[4]))
            loss_D_real.append(float(parts[6]))
            loss_D_fake.append(float(parts[8]))

            current_iteration = epoch * total_iteration + iteration
            iterations.append(current_iteration)

plt.figure(figsize=(10, 8))

loss_D_real_numpy = np.asarray(loss_D_real)
med_filtered_loss_D_real = scipy.signal.medfilt(loss_D_real_numpy, 51)
plt.subplot(2, 2, 1)
plt.plot(iterations, med_filtered_loss_D_real, label='loss_D_real')
plt.xlabel('Iterations')
plt.ylabel('loss_D_real')
plt.legend()

loss_D_fake_numpy = np.asarray(loss_D_fake)
med_filtered_loss_D_fake = scipy.signal.medfilt(loss_D_fake_numpy, 51)
plt.subplot(2, 2, 2)
plt.plot(iterations, med_filtered_loss_D_fake, label='loss_D_fake')
plt.xlabel('Iterations')
plt.ylabel('loss_D_fake')
plt.legend()

loss_D_numpy = np.asarray(loss_D)
med_filtered_loss_D = scipy.signal.medfilt(-loss_D_numpy, 51)
plt.subplot(2, 2, 3)
plt.plot(iterations, med_filtered_loss_D, label='loss_D')
plt.xlabel('Iterations')
plt.ylabel('loss_D')
plt.legend()

loss_G_numpy = np.asarray(loss_G)
med_filtered_loss_G = scipy.signal.medfilt(loss_G_numpy, 51)
plt.subplot(2, 2, 4)
plt.plot(iterations, med_filtered_loss_G, label='loss_G')
plt.xlabel('Iterations')
plt.ylabel('loss_G')
plt.legend()

plt.tight_layout()

plt.savefig(logger_save_path)
plt.close()
