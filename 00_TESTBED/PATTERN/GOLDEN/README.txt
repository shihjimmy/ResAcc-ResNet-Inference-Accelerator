Golden intermediate results from quant.py (one value per line, hex).
Tensor order: index = c*H*W + h*W + w. Signed values are two's complement.

stem_acc     shape (16, 16, 16)   32 bit signed
l1_a_acc     shape (16, 8, 8)     32 bit signed
l1_b_acc     shape (16, 8, 8)     32 bit signed
l1_sc_acc    shape (16, 8, 8)     32 bit signed
l2_a_acc     shape (32, 4, 4)     32 bit signed
l2_b_acc     shape (32, 4, 4)     32 bit signed
l2_sc_acc    shape (32, 4, 4)     32 bit signed
stem_out     shape (16, 16, 16)    8 bit unsigned
l1_a_out     shape (16, 8, 8)      8 bit unsigned
l1_out       shape (16, 8, 8)      8 bit unsigned
l2_a_out     shape (32, 4, 4)      8 bit unsigned
l2_out       shape (32, 4, 4)      8 bit unsigned
pool_sum     shape (32,)          12 bit unsigned
fc_logit     shape (10,)          32 bit signed
