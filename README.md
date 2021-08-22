# tensorflow
TensorFlow Tutorials

## Installation
1. Install VSCode and Python
    1. [VSCode](https://code.visualstudio.com/download)
    2. Python and its dependencies:
    ```text
    sudo apt update
    sudo apt upgrade
    sudo apt install pip

    pip install numpy
    pip install tensorflow  # or tensorflow-gpu
    pip install six
    pip install matplotlib
    pip install ipykernel
    ```
2. Install extensions from the VSCode:
    1. `Python`
    2. `Python for VSCode`
    3. `Python Extension Pack`
    4. `TensorFlow Snippets`
    5. `Jupyter`

## Introduction of TensorFlow
* TensorFlow Mechanics:
1. build <b>graph</b> using TensorFlow operations
2. feed <b>data</b> and run the <b>graph</b> (operation)

    `sess.run(op,feed_dict=Px:x_data)`

3. update variables in the <b>graph</b> and return <b>values</b>

### Tensor Ranks, Shapes, and Types
|Rank|Math entity|Python example|
|----|-----------|--------------|
|0   |Scalar (magnitude only)|`s = 483`|
|1   |Vector (magnitude and direction)|`v = [1.1, 2.2, 3.3]`|
|2   |Matrix (table of numbers)|`m = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]`|
|3   |3-Tensor (cube of numbers)|`t = [[[2], [4], [6]], [[8], [10], [12]], [[14], [16], [18]]]`|
|<i>n</i>|<i>n</i>-Tensor (you get the idea)|`...`|


|Rank|Shape|Dimension number|Example|
|----|-----------|--------------|---|
|0   |`[]`|0-D|A 0-D tensor. A Scalar.|
|1   |`[D0]`|1-D|A 1-D tensor with shape `[5]`.|
|2   |`[D0, D1]`|2-D|A 2-D tensor with shape `[3, 4]`.|
|3   |`[D0, D1, D2]`|3-D|A 3-D tensor with shape `[2, 3, 4]`.|
|<i>n</i>   |`[D0, D1, ..., Dn`]|<i>n</i>-D|A tensor with shape `[D0, D1, ..., Dn]`.|

|Data type|Python type|Description|
|---------|-----------|-----------|
|DT_FLOAT|`tf.float32`|32 bits floating point.|
|DT_DOUBLE|`tf.float64`|64 bits floating point.|
|DT_INT8|`tf.int8`|8 bits signed integer.|
|DT_INT16|`tf.int16`|16 bits signed integer.|
|DT_INT32|`tf.int32`|32 bits signed integer.|
|DT_INT64|`tf.int64`|64 bits signed integer.|
|...|...|...|

### Epoch, Batch Size, Number of Iterations
* To be clear, one pass is one forward pass + one backward pass.
* One epoch
    * One forward pass and one backward pass of all the training examples
* Batch size
    * The number of training examples in one forward/backward pass.
    * The higher the batch size, the more memory space need.
* Number of iterations
    * Number of passes, each pass using number of examples(batch size)

* 1,000 training examples, and batch size is 500, then it will take 2 iterations to complete 1 epoch.

[Module: tf.dtypes](https://www.tensorflow.org/api_docs/python/tf/dtypes)

## Useful Website:
* [desmos](https://www.desmos.com/calculator?lang=ko)