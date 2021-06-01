# # https://fairyonice.github.io/Visualization%20of%20Filters%20with%20Keras.html
# layer_output = layer_dict["predictions"].output
#
out_index = [65, 18, 92, 721]
for i in out_index:
    visualizer = VisualizeImageMaximizeFmap(pic_shape=(224, 224, 3))
    images = []
    probs = []
    myprob = 0
    n_alg = 0
    while (myprob < 0.9):
        myimage = visualizer.find_images_for_layer(input_img, layer_output, [i],
                                                   picorig=True, n_iter=20)
        y_pred = model.predict(myimage[0][0]).flatten()
        myprob = y_pred[i]
        n_alg += 1

    print("The total number of times the gradient ascent needs to run: {}".format(n_alg))

    argimage = {"prediction": [myimage]}
    print("{} probability:".format(classlabel[i])),
    print("{:4.3}".format(myprob)),

    visualizer.plot_images_wrapper(argimage, n_row=1, scale=4)
