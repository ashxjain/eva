##Assignment-1B

### What are Channels and Kernels (according to EVA)?

Channels are similar feature bags. For example, a RGB image has 3 channels for each color. Post convolution, the image contains as many channels as the kernel. Usually when there are lots of features in the input channel, kernel is set to have high channels to group all the similar features. Like for MNIST database, all vertical edges will be together in one channel, all horizontal edges in another channel etc.
Kernels are a simple matrix which convolves on the input image to extract features. For example, a horizontal edge kernel will extract all the edges from the input image. 

Channels and kernels go hand in hand. As kernels extracts the features, channels groups the similar features.

### Why should we only (well mostly) use 3x3 Kernels?

Usage of 3x3 kernels leads to less parameters and hence making it more computation efficient. Higher dimensions can be modeled with 3x3 kernel with lesser parameters. For example a 5x5 kernel can be achieved by using two 3x3 kernel. If we compare number of parameters for a 5x5 kernel and two 3x3 kernel:

One 5x5 kernel : $1*5*5=25$ parameters
Two 3x3 kernel : $2*3*3=18$ parameters

It can be observed that the number of parameters reduces and hence saves lot of computations. Following  [plot](https://goo.gl/d6RAaW) shows the number of parameters with respective to input image of dimension nxn:

![plot_3x3_vs_5x5](https://goo.gl/4NMZ8P)

It can also be inferred that with increased input size, two 3x3 kernel performs better than a single 5x5 kernel. Similarly, a 7x7 kernel can be replaced with three 3x3 kernel.

Also with 3x3 kernel, we get more number of layers and hence we can capture more complex features.

### How many times do we need to perform 3x3 convolution operation to reach 1x1 from 199x199 (show calculations)

99 Times
```
199x199 | 197x197 | 195x195 | 193x193 | 191x191 | 189x189 = 5
189x189 | 187x187 | 185x185 | 183x183 | 181x181 | 179x179 = 5
179x179 | 177x177 | 175x175 | 173x173 | 171x171 | 169x169 = 5
169x169 | 167x167 | 165x165 | 163x163 | 161x161 | 159x159 = 5
159x159 | 157x157 | 155x155 | 153x153 | 151x151 | 149x149 = 5
149x149 | 147x147 | 145x145 | 143x143 | 141x141 | 139x139 = 5
139x139 | 137x137 | 135x135 | 133x133 | 131x131 | 129x129 = 5
129x129 | 127x127 | 125x125 | 123x123 | 121x121 | 119x119 = 5
119x119 | 117x117 | 115x115 | 113x113 | 111x111 | 109x109 = 5
109x109 | 107x107 | 105x105 | 103x103 | 101x101 | 99x99 = 5
99x99 | 97x97 | 95x95 | 93x93 | 91x91 | 89x89 = 5
89x89 | 87x87 | 85x85 | 83x83 | 81x81 | 79x79 = 5
79x79 | 77x77 | 75x75 | 73x73 | 71x71 | 69x69 = 5
69x69 | 67x67 | 65x65 | 63x63 | 61x61 | 59x59 = 5
59x59 | 57x57 | 55x55 | 53x53 | 51x51 | 49x49 = 5
49x49 | 47x47 | 45x45 | 43x43 | 41x41 | 39x39 = 5
39x39 | 37x37 | 35x35 | 33x33 | 31x31 | 29x29 = 5
29x29 | 27x27 | 25x25 | 23x23 | 21x21 | 19x19 = 5
19x19 | 17x17 | 15x15 | 13x13 | 11x11 | 9x9 = 5
9x9 | 7x7 | 5x5 | 3x3 | 1x1 = 4
Total = 5*19 + 4*1 = 99 times
```
