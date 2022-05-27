#Imports 
from keras.models import *
from keras.layers import *
from keras.utils.vis_utils import plot_model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

from matplotlib import pyplot
from matplotlib.image import imread
from numpy import *
from numpy.random import *
import numpy as np
import sys

sys.path.append('..')

#Import Data and Pre-Process

####################################################################################################
#Folders
folderURLs = np.array(["Pokemon_Image_Dataset/yellow/", "Pokemon_Image_Dataset/red-blue/", 
                      "Pokemon_Image_Dataset/red-green/", "Pokemon_Image_Dataset/silver/",
                      "Pokemon_Image_Dataset/gold/", "Pokemon_Image_Dataset/crystal/", "Pokemon_Image_Dataset/firered-leafgreen/",
                      "Pokemon_Image_Dataset/ruby-sapphire/", "Pokemon_Image_Dataset/emerald/",
                      "Pokemon_Image_Dataset/diamond-pearl/", "Pokemon_Image_Dataset/platinum/",
                      "Pokemon_Image_Dataset/black-white/", "Pokemon_Image_Dataset/heartgold-soulsilver/"])

folderImgLastIndex = np.array([151, 151, 151, 251, 251, 251, 389, 387, 388, 512, 514, 750, 572])
folderCount = len(folderURLs)

####################################################################################################
#Import Data
#def whiteBackground(image):
#    for i in range(64):
#        for j in range(64):
#            if image[i,j,3] == 0.0:
#                image[i,j,0] = 0.0
#                image[i,j,1] = 0.0
#                image[i,j,2] = 0.0
#    return image

pokemonData_All = []

for folder in range(6, 13):
    #newImgNum = 1
    for img in range(1, folderImgLastIndex[folder] + 1):
        try:
            imgURL = folderURLs[folder] + str(img)
            image = imread(imgURL, format="png")
            
            #img = Image.open(imgURL)
            #image = whiteBackground(image)
            #img = img.resize((64,64))
            
            #saveImgURL = folderURLs[folder] + str(newImgNum)
            #newImgNum += 1
            #img = img.save(saveImgURL, format="png")
            
            pokemonData_All.append(image)
        except:
            error = "No Image Found"

pokemonData_All = np.asarray(pokemonData_All)
pokemonData_All = pokemonData_All[:,:,:,0:3]
n_samples, x_size, y_size, n_channels = pokemonData_All.shape

####################################################################################################
#Shuffle Then Plot Images From The Dataset
np.random.shuffle(pokemonData_All)
for i in range(25):
    pyplot.subplot(5, 5, 1 + i)
    pyplot.axis('off')
    pyplot.imshow(pokemonData_All[i])
    
print(pokemonData_All.shape)
pyplot.savefig("test.png")
pyplot.show()

#Functions/ Define Models

#################################################################################################### 
#Building Discriminator Model
def define_Discriminator(in_shape=(64,64,3)):
    initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
    model = Sequential()
    model.add(Conv2D(64, kernel_size=4, strides=2, padding='same', kernel_initializer=initializer, input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Conv2D(128, kernel_size=4, strides=2, padding='same', kernel_initializer=initializer))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(128, kernel_size=4, strides=2, padding='same', kernel_initializer=initializer))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Flatten())
    #model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    
    #Compile model
    opt = Adam(learning_rate=0.0001, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

#################################################################################################### 
#Building Generator Model
def define_Generator(latent_dim):
    initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
    model = Sequential()
    
    n_nodes = 256 * 8 * 8
    model.add(Dense(n_nodes, input_dim=latent_dim, kernel_initializer=initializer))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((8, 8, 256)))
    
    model.add(Conv2DTranspose(256, kernel_size=4, strides=2, padding='same', kernel_initializer=initializer))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Conv2DTranspose(256, kernel_size=4, strides=2, padding='same', kernel_initializer=initializer))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Conv2DTranspose(256, kernel_size=4, strides=2, padding='same', kernel_initializer=initializer))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Conv2D(3, kernel_size=5, activation='sigmoid', padding='same', kernel_initializer=initializer))
    return model

#################################################################################################### 
#Building GAN Model From Discriminator And Generator Models
def define_gan(g_model, d_model):
    #Make Weights In The Discriminator Not Trainable
    d_model.trainable = False
    #Connect Them
    model = Sequential()
    #Add Generator
    model.add(g_model)
    #Add The Discriminator
    model.add(d_model)
    #Compile Model
    opt = Adam(learning_rate=0.0001, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model

#Auxiliary Functions

#################################################################################################### 
#Functions For Discriminator Model
def load_real_samples(dataset): 
    return dataset

def generate_real_samples(dataset, n_batch):
    #Choose Random Instances
    ix = randint(0, dataset.shape[0], n_batch)
    #Retrieve Selected Images
    X = dataset[ix]
    #Generate 'Real' Class Labels (1)
    y = ones((n_batch, 1))
    return X, y

def d_generate_fake_samples(n_batch):
    # generate uniform random numbers in [0,1]
    X = rand(n_batch * 64 * 64 * 3) 
    
    X = X.reshape((n_batch, 64, 64, 3))
    X = (X - 0.5)/0.5
    # generate 'fake' class labels (0)
    y = zeros((n_batch, 1))
    return X, y

def train_discriminator(model, dataset, n_iter=100, n_batch=32):
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    for i in range(n_iter):
        # get randomly selected 'real' samples
        X_real, y_real = generate_real_samples(dataset, half_batch)
        
        # update discriminator on real samples
        _, real_acc = model.train_on_batch(X_real, y_real)
        
        # generate 'fake' examples
        X_fake, y_fake = d_generate_fake_samples(half_batch)
        
        # update discriminator on fake samples
        _, fake_acc = model.train_on_batch(X_fake, y_fake)
        
        # summarize performance
        print('>%d real=%.0f%% fake=%.0f%%' % (i+1, real_acc*100, fake_acc*100))
        
####################################################################################################        
#Functions For Generator Model
def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input

def g_generate_fake_samples(g_model, latent_dim, n_samples):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    X = g_model.predict(x_input)
    # create 'fake' class labels (0)
    y = zeros((n_samples, 1))
    return X, y

#################################################################################################### 
#Train GAN Save Model And Plots
def save_plot(examples, epoch, n=5):
    # plot images
    for i in range(n * n):
        # define subplot
        pyplot.subplot(n, n, 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(examples[i])
    # save plot to file
    filename = 'generated_plot_e%03d.png' % (epoch+1)
    pyplot.savefig(filename)
    pyplot.close()

#Evaluate The Discriminator, Plot Generated Images, Save Generator Model
def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=200):
    # prepare real samples
    X_real, y_real = generate_real_samples(dataset, n_samples)
    # evaluate discriminator on real examples
    _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
    # prepare fake examples
    x_fake, y_fake = g_generate_fake_samples(g_model, latent_dim, n_samples)
    #x_fake = (x_fake + 1) / 2.0
    # evaluate discriminator on fake examples
    _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
    # summarize discriminator performance
    print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
    # save plot
    save_plot(x_fake, epoch)
    print(x_fake.shape)
    # save the generator model tile file
    filename = 'generator_model_%03d.h5' % (epoch + 1)
    g_model.save(filename)
    
#Train The Generator And Discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=15000, n_batch=500):
    bat_per_epo = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    for i in range(n_epochs):
        # enumerate batches over the training set
        for j in range(bat_per_epo):
            # get randomly selected 'real' samples
            X_real, y_real = generate_real_samples(dataset, half_batch)
            # generate 'fake' examples
            X_fake, y_fake = g_generate_fake_samples(g_model, latent_dim, half_batch)
            
            # create training set for the discriminator
            X, y = vstack((X_real, X_fake)), vstack((y_real, y_fake))
            
            # update discriminator model weights
            d_loss, _ = d_model.train_on_batch(X, y)
            
            # prepare points in latent space as input for the generator
            X_gan = generate_latent_points(latent_dim, n_batch)
            
            # create inverted labels for the fake samples
            y_gan = ones((n_batch, 1))
            
            # update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            # summarize loss on this batch
            print('>%d, %d/%d, d=%.3f, g=%.3f' % (i+1, j+1, bat_per_epo, d_loss, g_loss))
        # evaluate the model performance, sometimes
        if (i+1) % 10 == 0:
            summarize_performance(i, g_model, d_model, dataset, latent_dim)
            
            #Model Summaries

####################################################################################################
#Define Discriminator Model
d_model = define_Discriminator()

#Summarize Discriminator Model
d_model.summary()

####################################################################################################
#Define The Size Of The latent Space
latent_dim = 50

#Define Generator Model
g_model = define_Generator(latent_dim)

#Summarize Generator Model
g_model.summary()

#Generate Latent Samples
n_latent_samples = 9
X, _ = g_generate_fake_samples(g_model, latent_dim, n_latent_samples)
print(X.shape)
#X = (X + 1) / 2.0
#Plot The Generated Samples
for i in range(n_latent_samples):
    #Define Subplot
    pyplot.subplot(3, 3, 1 + i)
    #Turn Off Axis Labels
    pyplot.axis('off')
    #Plot Single Image
    pyplot.imshow(X[i])
    
#Show The Samples
pyplot.show()

#Training And Output

####################################################################################################
#Retrieve Our Training Data Set Which Has Been Normalized Between (0-1)
dataset = load_real_samples(pokemonData_All)

# #Train The Discriminator Model On train_X Data
train_discriminator(d_model, dataset)

# #Define GAN Model
gan_model = define_gan(g_model, d_model)

# #Train GAN Model
train(g_model, d_model, gan_model, dataset, latent_dim)