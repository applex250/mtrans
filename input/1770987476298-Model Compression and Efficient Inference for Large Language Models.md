# Model Compression and Efficient Inference forLarge Language Models: A Survey

Wenxiao Wang†, Wei Chen†, Yicong Luo†, Yongliu Long†, Zhengkai Lin†, Liye Zhang†, Binbin Lin, DengCai, Xiaofei He Senior Member, IEEE

Abstract—Transformer based large language models have achieved tremendous success. However, the significant memory andcomputational costs incurred during the inference process make it challenging to deploy large models on resource-constraineddevices. In this paper, we investigate compression and efficient inference methods for large language models from an algorithmicperspective. Regarding taxonomy, similar to smaller models, compression and acceleration algorithms for large language models canstill be categorized into quantization, pruning, distillation, compact architecture design, dynamic networks. However, Large languagemodels have two prominent characteristics compared to smaller models: (1) Most of compression algorithms require finetuning or evenretraining the model after compression. The most notable aspect of large models is the very high cost associated with model finetuningor training. Therefore, many algorithms for large models, such as quantization and pruning, start to explore tuning-free algorithms. (2)Large models emphasize versatility and generalization rather than performance on a single task. Hence, many algorithms, such asknowledge distillation, focus on how to preserving their versatility and generalization after compression. Since these two characteristicswere not very pronounced in early large models, we further distinguish large language models into medium models and “real” largemodels. Additionally, we also provide an introduction to some mature frameworks for efficient inference of large models, which cansupport basic compression or acceleration algorithms, greatly facilitating model deployment for users.

Index Terms—Large language models, model compression, efficient inference, quantization, pruning, knowledge distillation, compactarchitecture design, dynamic networks.

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-11/0a775777-6cdd-4dbe-a304-295f64b9a1ad/d6a379a0cde44c9350c2e333a18f8f2023f2ca1a332bdf48121c0a067ca7333d.jpg)


# 1 INTRODUCTION

ARGE language models (LLMs) has become an im-L portant and popular topic in the artificial intelligencefield. Compared with previous language models, LLMs (e.g.,ChatGPT, LLaMA, Claude) show much greater generaliza-tion capability for their unseen data. Furthermore, they evenpresent many abilities that smaller models do not present(i.e., emergent abilities), such as multi-step reasoning and in-struction following abilities. These progresses demonstrategreat potentials of LLMs.

However, the forbidding memory and computationalbudgets in the inference process also prevent the deploy-ment of LLMs. For example, a 10B models with float32weights consumes 37GB memory, needless to say that the in-ference memory cost will further increase in a speed squareto the sequence length. To deploy the models on resourceconstrained devices, or even mobile devices, many LLMsresort to model compression methods such as quantizationto reduce the inference memory and computational cost.

Model compression for deep learning models is a fieldthat appear much earlier than LLMs. It assumes that wehave already a pre-defined (or even pretrained) model.Model compression devotes to reducing the memory and

computational cost of the model in the inference process, sothat the model can run on various resource-constrained de-vices. Algorithmically, common model compression meth-ods include:

Quantization transforms float32 weights or activa-tions into lower-bit float or integer numbers. Less bitsmeans less memory requirement. Further, less bitsmay indicate higher parallelism and faster inferencespeed.

Pruning devotes to removing unimportant compo-nents (e.g., neurons, layers, etc) in a pre-designedmodel, thus reducing the memory and computa-tional cost in the inference cost.

Knowledge distillation introduces a pretrainedlarge model as a teacher and transfers its knowledgeto a new smaller model which is called a studentmodel. Then, the smaller model will share nearly thesame ability as the teacher and enjoy less memoryand computational cost.

Compact architecture design designs new operatorswith less cost to replace (often approximate) thecumbersome operators in the original model. Forthe Transformer models, self-attentions are the maintargets and are often replaced with other operators.

Dynamic networks treat each inference sample dif-ferently. The original model is a super-net, and eachsample only selects a sub-structure of the super-netfor inference. Mixture of experts (MoE) is a kind ofdynamic inference.

Besides, the above methods can also be combined for

further compression and speedup. Existing compressionmethods have provided us with important cornerstonesand insights to compress LLMs. However, LLMs also bringmany new challenges for model compression:

1) Many previous model compression methods of-ten require to finetuning models after compres-sion. However, since the great budget to finetuningLLMs, researchers have to explore finetuning freeor, at least, more efficient finetuning methods.

2) Instead of handling one single task such as neuralmachine translation, large language models emph-size versatility and generalization across varioustasks and unseen data, or even emergent abilities.Thus, large language models after compression re-quire more careful validation of their versatility andgeneralization.

To face these challenges, many compression methodsspecialized for LLMs are proposed. In this paper, we willgive a comprehensive survey about these methods. To betterpresent these methods, we further isolate language modelsaround one billion or less parameters, such as BERT, GPT-2, and call them medium models, though they are usuallytaken as large language models. And Models with overone billion parameters, such as LLaMA, Claude, ChatGPT,and etc, keep the name of large language models. Thereasons is that medium models are less affected by the abovetwo challenges, i.e., medium models are relatively easy tofinetune, demonstrate fewer emergent abilities. As a result,many compression methods for medium models are stillsimilar to those for smaller models.

The following sections are organized as: Some prelimi-naries will be introduced in Section 2. Then, we will discusspruning, knowledge distillation, quantization, compact ar-chitecture design and dynamic networks in Section 3, 4, 5,6, 7, 8, respectively.

# 2 PRELIMINARIES

In this section, we will introduce some essential preliminar-ies about Transformer, large language models, parameter-efficient training, and etc.

# 2.1 Transformer

Transformer is first introduced in [1], which is employedfor machine translation initially. Its fundamental structure isdepicted in Fig. 1. The input (a sentence) is often projectedthrough an embedding layer to a sequence of vectors (calledtokens) for a Transformer’s input. Each Transformer blockconsists of an attention module and a multi-layer preceptron(MLP) module.

Attention. For each token in the input sequence, it is firstmapped (often with a linear function) into vectors of query$( Q )$ and/or key-value pairs ( $K$ and $V$ ). Then, the attentionmodule can be described as mapping a query and a set ofkey-value pairs to an output. The output is computed asa weighted sum of the values, where the weight assignedto each value is computed by a compatibility function of

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-11/0a775777-6cdd-4dbe-a304-295f64b9a1ad/3c292e626d681ac22535dca51b01d5c1ca7316f31c7359cd34f742e7be4e72ca.jpg)



Fig. 1: The Transformer architecture drawn from [1].


the query with the corresponding key. The most commonattention module is scaled dot-product function:

$$
\operatorname {A t t e n t i o n} (Q, K, V) = \operatorname {s o f t m a x} \left(\frac {Q K ^ {T}}{\sqrt {d _ {k}}}\right) V, \tag {1}
$$

where the weight is computed through the dot-product of√$Q$ and $K ,$ and $\sqrt { d _ { k } }$ is a constant scaling factor.

Multi-head Attention. Further, instead of performinga single attention function with keys, values and queries,Transformer employs a multi-head attention [1], as shown inFig. 1. It maps input tokens into $h$ distinct queries, keys andvalues $( \{ Q _ { i } , K _ { i } , \bar { V } _ { i } | i \in [ 1 , h ] \} )$ with different linear layers.Then, the final output becomes:

$$
\text {M u t i - H e a d A t t e n t i o n} = \operatorname {C o n c a t} \left(\operatorname {h e a d} _ {1}, \dots , \operatorname {h e a d} _ {\mathrm {h}}\right) W _ {o}
$$

$$
\operatorname {h e a d} _ {\mathrm {i}} = \operatorname {A t t e n t i o n} \left(Q _ {i}, K _ {i}, V _ {i}\right), \tag {2}
$$

where $W _ { o }$ is a linear projection matrix.

Encoder and docoder. The first intial Transformer isfor neural machine translation, which employs an encoder-decoder structure. The encoder first handles input sequnce(e.g., wirtten in the source language) independently, and thedecoder takes the encoder’s output as input and predicts thefinal output (e.g., the target language). There are two coredifferences between an encoder’s and a decoder’s attentionmodule: (1) The encoder employs a full attention, whereany two tokens in the input sequence are visible to eachother. On the other hand, the decoder employs a single-direction attention. The reason is the decoder generates out-put tokens one by one, and each token can only seen outputtokens before itself. (2) The encoder employs a self-attentionmodule, that is, $Q , K , V$ are all from input tokens of thesource language. In contrast, the decoder employs a cross-attention, where $K , V$ are from the encoder’s output, while$Q$ is last output token of the decoder. As the development, in

addition to the encoder-decoder models (e.g., T5 [2]), plentyof following language models also employ pure encoderstructure (e.g., BERT [3]) and pure decoder structure suchas GPT-series [4], [5], [6] models.

While we brifly introduce some important concepts inTransformer, more subtle introduction can be seen in manyprevious surveys [7], [8].

# 2.2 Medium/Large Language Models

As the success of Transformer, more and more pure Trans-former based language models emerge, and the parame-ters of models also increase. Though there is no specificthreshold for large language models’ scale of parameter, itis commonly accepted that “large” language models canbe dated back from BERT [3] and GPT-1 [4], which areboth proposed in 2018. Their scales of parameter both reachseveral hundred million.

After that, more language models such as GPT-3 [9],PanGu [10], T5 [2], CPM-2 [11], BLOOM [12], OPT [13],GLM [14], PaLM [15], QWen [16], ERNIE [17], LLaMA [18],and etcare proposed. Besides scale of parameter, the mostsignificant property that distinguish these models from theprevious are their emergence. As proposed in [19], largelanguage models utilize large-scale self-supervised pretrain-ing to enable the models with many abilities (i.e., emergentabilities) that do not appear in smaller models, includingmulti-step reasoning, instruction following, program exe-cution abilities, and etc. For example, GPT-3, LLaMA andmany other LLMs can solve few-shot tasks through in-context learning, and even zero-shot tasks. The breakoutof large language models present the surprising abilities(called emergence) in solving a series of complex tasks oversmaller models.

To further emphasize this difference, we catorgorizelanguage models over hundred millions of parameters intomedium models and large models. Specifically, models witharound one billion or less parameters are called mediummodels, while those with over one billion parameters arecalled large models.

# 2.3 Parameter-efficient Finetuning (PEFT)

As we discussed in the introduction, many model compres-sion algorithms such as knowledge distillation and pruningrequire finetuning or even training for accuracy recoveryafter compression. Nevertheless, full-parameter finetuningor training is very costly for medium or large models. To thisend, many parameter efficient finetuning (PEFT) algorithmsare proposed. They devote to finetune as few parametersor epochs as possible to lower the finetuning cost. In thispaper, model compression and acceleration algorithms inthe inference stage (rather than training/finetuning stage)are mainly discussed, but we still supplement some PEFTalgorithms in the Appendix A.

# 3 QUANTIZATION

Quantization refers to the process of mapping input valuesin a large (often continuous) set to output values in a small(often finite) set (see Fig. 2 for an example). It is the moststraightforward method to cut down memory cost and

improve inference speed for LLMs, especially on hardwarewith support for fast operation of low-bit datatypes (e.g.,INT4). It should be noted that quantization has achievedimpressive success in both neural network training [20] andinference, while the focus of this survey is only the inferencepart.

Quantization methods have several advantages overother compression methods, such as pruning and distilla-tion. 1) High compression ratio: quantizing the weights inLLMs from 32-bit float to 4-bit integer could drasticallycompress the model size to approximately 1/8, essential formemory-bound1 processes like LLM inference. 2) Low cost:a bunch of quantization methods doesn’t require re-trainingthe entire LLMs, making it more affordable to researcherswith limited computing resources. 3) High flexibility: quanti-zation is compatible with most other compression methods,which introduces exceptional opportunities for further im-proving the performance.

To help readers better understand quantization methods,we will first introduce the standard quantization methodand some basic concepts in Subsection 3.1. Then, in Section3.2, we will briefly summarize some of the most importantworks for medium-size Language models (e.g., BERT, GPT-2, etc.) before the emergence of LLMs. Section 3.3 andSection 3.4 covers recent advances in quantization methodsthat focus on LLMs inference. Considering the pains in re-training a model with tens of billions of parameters, wegenerally divide LLM quantization methods into two partsbased on whether the technique needs re-training. Methodswithout re-training (i.e., post-training quantization, PTQ) arediscussed in Section 3.3 while methods that demand re-training (i.e., quantization-aware training, QAT) is discussed inSection 3.4. Finally, in Section 3.5, we discuss some advancedtopics showing potential for future research but not coveredin previous sections.

# 3.1 Basic Concepts

Quantization has a history that is much longer than neuralnetworks, and specific quantization methods vary a lot. Togive readers a clear grasp of the diverse quantization con-cepts, we will first introduce the standard uniform quanti-zation and the corresponding dequantization process. Afterthat, we explain several fundamental concepts frequentlymentioned in different quantization methods.

1) Uniform quantization. The most basic form of quanti-zation is to separate a real-valued range into uniform, finiteintervals (e.g., $2 ^ { b }$ intervals for $b$ -bit integer) and then mapreal values within the same interval to the same integer. Theformula of such a process is as follows:

$$
Q (r) = \operatorname {R O U N D} \left(\frac {r}{S}\right) + Z \tag {3}
$$

where $Q ( \cdot )$ is the quantization operator, $r$ is the real valuesto be quantized, $S$ is a real valued scaling factor, $Z$ is aninteger zero point and ROUND(·) is a rounding operation(e.g., round to nearest). This method is known as uniform

1. ”memory-bound” means that the transfer between the device andglobal memory nearly reaches the limitation or fetching data from thememory is the bottleneck of the whole process. On the contrary, the”compute-bound” process spends most of its time calculating and notaccessing memory.

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-11/0a775777-6cdd-4dbe-a304-295f64b9a1ad/9a4486a0c2d480a77568eaadca2c15e7b48920ae4f64ddc254ad3c86b658bcd8.jpg)



(a) Illustration of uniform quantization process



FP16 Tensor


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-11/0a775777-6cdd-4dbe-a304-295f64b9a1ad/7da523c69eea5d9ea748776acf447a23215c0edaf678aded76b4aebf2aa3aafc.jpg)



(b) Quantization and De-quantization of a FP16 tensor



Fig. 2: (a) Uniform quantization separates a real-valuedrange into uniform, finite intervals and then maps real valueswithin the same interval to the same integer. (b) An FP16tensor is quantized into INT4 format and then dequantizedback into FP16.


quantization since the length of each interval is the same(equal to scale factor $S$ ) and the quantized values areuniformly spaced (e.g., integers $0 , 1 , 2 , \ldots )$ .

The operation to recover real values from quantizedvalues is called dequantization:

$$
\tilde {r} = S \cdot (Q (r) - Z) \tag {4}
$$

It should be noted that the recovered values $\tilde { r }$ may be dif-ferent from the original real values $r$ due to the informationloss introduced by ROUND(·) function.

2) Non-uniform quantization. The counterpart of uniformquantization is non-uniform quantization, where quantizedvalues are not necessarily uniformly spaced, and the lengthof intervals can be different. The general formula for non-uniform quantization is:

$$
Q (r) = Q _ {i}, \text {i f} r \in [ \Delta_ {i}, \Delta_ {i + 1}) \tag {5}
$$

where $Q _ { i }$ is the candidate quantized values called quanti-zation levels, $\Delta _ { i }$ and $\Delta _ { i + 1 }$ defines an interval in which realvalues would be mapped to corresponding $Q _ { i }$ .

Given a fixed bit-width, non-uniform quantization canoften achieve higher accuracy and lower quantization er-ror than its uniform counterpart, as the weights of neuralnetworks are generally not uniformly distributed. How-ever, non-uniform methods may suffer low computationefficiency because these methods usually involve a time-consuming lookup operation that is not directly compatiblewith parallel computation hardware like GPU.

3) Clipping range and calibration. An important factorassociated with uniform quantization is clipping range $[ \alpha , \beta ]$so that real values lower than $\alpha$ or higher than $\beta$ would

be clipped to $\alpha$ or $\beta$ respectively. The clipping range alsodirectly influences the scaling factor $S$ in uniform quantiza-tion:

$$
S = \frac {\beta - \alpha}{2 ^ {b} - 1} \tag {6}
$$

In general, a wider clipping range results in fewer outliersrequiring clipping in the input data. However, this comesat the cost of a larger scale factor, as shown in Equation (6).Consequently, the quantization error for real values withinthe clipping range would also be larger.

Choosing the clipping range is referred to as calibration.Common choices of calibration involve using min/maxvalues (i.e., $- \alpha = r _ { m i n } , \beta = r _ { m a x } )$ , using absolute maxvalues (i.e., $- \alpha = \beta = m a x ( | r | ) )$ or minimizing the infor-mation loss (i.e., KL divergence) between the real values andquantized values.

4) Symmetric/Asymmetric quantization. When the clippingrange $[ \alpha , \beta ]$ is symmetric with respect to 0 $( \alpha + \beta = 0$ and$Z = 0$ ), then corresponding method is often referred to assymmetric quantization; otherwise asymmetric quantization.

The clipping range for asymmetric quantization ismore compact, which is especially important for activa-tions in neural networks whose range may be signifi-cantly imbalanced (e.g., after ReLU, all activations becomenon-negative). Symmetric quantization, however, is morecompute-efficient at the dequantization step.

5) Quantization Granularity. A categorization criterionspecified for quantization of neural network is the granu-larity, which corresponds to which weights/activations arequantized together and share quantization parameters. Welist typical granularity from coarse to fine as follows:

Layer-wise: Weights of all filters in a layer for con-volution layers or the full weight matrix for linearlayers are quantized together.

Channel-wise: Weights of a single filter for convolu-tion layers are quantized together.

Row/Column-wise: Weights of a single row/columnof weight matrix for linear layers are quantized to-gether.

Token-wise: Activations for each token are quantizedtogether.

Group-wise: Several consecutive real values inweights or activations are viewed as a group andquantized together. The group size is typically small(e.g., 64 consecutive real values).

In general, finer granularity divides model weights or acti-vations into smaller groups, which can reduce quantizationerrors. However, finer granularity requires storing morequantization parameters and introduces higher quantizationoverhead during computation.

Note that different works may use different notations.For clarity, in this survey, we set input $\mathbf { X } \in \mathbb { R } ^ { N \times D _ { \mathrm { i n } } }$ , weightmatrix $\dot { \mathbf { W } } \in \mathbb { R } ^ { D _ { \mathrm { i n } } \times D _ { \mathrm { o u t } } }$ , where $N$ is the batch size, $D _ { \mathrm { i n } }$ and$D _ { \mathrm { o u t } }$ are input and output dimensions respectively. A linearlayer can be written as $\mathbf { Y } = \mathbf { X } \mathbf { W } \in \mathbb { R } ^ { N \times \hat { D } _ { \mathrm { o u t } } }$ .

6) Post-Training Quantization/Quantization-Aware Train-ing. An effective way to reduce quantization error isthrough training. Quantization methods can either requirere-training after quantization or not:

Post-Training Quantization, PTQ: methods without re-training, quantized models can be directly used ininference.

Quantization-Aware Training, QAT: methods with re-training, which helps recover the error introduced byquantization.

Perturbation to parameters of a trained model caused byquantization may push the model away from the pointwhere it had converged in floating-point-precision train-ing. QAT addresses this issue through either simulatingthe quantization process in re-training (e.g., inject nodes inthe computation graph that simulate weight quantization)or using additional parameters to finetune the quantizedmodel (e.g., combined with Adapters or LoRA) so that themodel can learn to converge to a point which will have abetter performance after quantization. However, re-trainingof full LLM with billions of parameters has an unbearablecost to most researchers. Hence, PTQ methods are alsoextensively studied to achieve better performance withoutintroducing additional computation budgets through spe-cific non-training methods (e.g., Optimal Brain Surgery).

Generally speaking, the rounding operation in quanti-zation is a non-differentiable process, so QAT may seemimpossible at first sight. However, researchers found thata simple method called Straight Through Estimator (STE)works well under most circumstances, ignoring the round-ing operation and approximating it with an identity func-tion.

7) Static/Dynamic quantization. A key difference betweenquantizing the weights and activations of a neural networklies in the fact that weights are mostly fixed during inferenceso that the statistics used in clipping range calibration canbe computed statically. It’s not the case for activations sinceits range and statistics are unknown until runtime. Thus,activation quantization can be divided into two categories.

Static quantization refers to the methods whose quan-tization parameters are pre-calculated before infer-ence using some calibration inputs to find typicalactivation statistics or learned jointly during neuralnetwork training.

Dynamic quantization calculates quantization parame-ters dynamically at runtime, which is generally moreaccurate than its static counterpart but can have highoverhead for calculating required statistics (e.g., min,max, etc.).

8) Simulated/Integer-Only quantization. Still, another cat-egory criterion is whether quantized values are used foractual operations (e.g., matrix multiplication). For simulatedquantization (also called fake quantization), the weights andactivations are only stored in low-bit datatypes (e.g., INT4)and have to be dequantized to high-bit datatypes (e.g., float16)to carry out actual operations. For Integer-only quantization,the operations can be carried out using low-bit datatypes.Simulated quantization can reduce the memory cost anddata-moving time of neural networks, which is helpful sinceseveral works have shown that LLM inference is memory-bound rather than compute-bound1. In contrast, integer-onlyquantization can further enjoy the acceleration of efficientlow-bit operations supported by specific hardware.

9) Weight-only/Weight $^ +$ Activation quantization. Whetherthe quantization objective is only weights or both weightsand activations. Previous work [21] found that activationquantization is generally more susceptible to weight quan-tization, so weight-only quantization can reach lower bit-width. However, quantized weights must be dequantizedbefore multiplying with activations, so weight-only quanti-zation can not be integer-only and will introduce additionalcomputation overhead during inference.

We’ve briefly covered some of the most essential con-cepts in quantization. These concepts are universally ap-plied to all neural network quantization, and each specificmethod may suit several different concepts (e.g., a uniformsymmetric dynamic layer-wise simulated quantization methodfor LLMs). We categorize the main quantization methods forLLMs according to these basic concepts in TABLE 1.

# 3.2 Quantization Methods for Medium-Size LanguageModels

For ease of expression, we refer to models with sizes smallerthan or close to 1B as medium-size language models, repre-sented by BERT, GPT-2, and BART.

Quantization methods for medium-size language mod-els [22] mainly adopt the QAT framework instead of PTQ,as the cost of re-training such a model is relatively accept-able. The improvement in evaluation metric (e.g., accuracy)brought by re-training is significant, especially under ex-treme low-bit settings (e.g., 1-bit or 2-bit quantization). As aresult, we will first introduce mainstream methods, i.e., QATmethods, for medium-size language models and then coverthe PTQ methods.

# 3.2.1 QAT for Medium-Size Language Models

Early works aim at quantizing weights of BERT-like modelsinto INT8. Q8BERT [23] applies the basic QAT frameworkfrom [24] to quantize both weights and activations of BERTinto 8-bits without significant reduction in model perfor-mance.

Some works enable quantization into bit-width lowerthan 8-bit using more complicated methods [25], [26], [27],[28], [29], [30], [31]. For example, Q-BERT [25] maintains8-bit activations and mixed-precision weights down to 2/3-bits. It uses the Hessian matrix to determine the bit-widthfor weights of each layer so that more aggressive quantiza-tion is performed for layers with smaller top eigenvaluesof the corresponding Hessian. Further, TernaryBERT [27]restricts its weights to $\cdot 1 , 0 , + 1 ,$ using only 2 bits, and em-ploys 8-bit activations. Knowledge distillation is adopted toovercome performance degradation by minimizing the sumof mean-square error (MSE) of the activations and attentionscores between the original and quantized model. FollowingTernaryBERT, BinaryBERT [32] pushes the limit of BERTquantization to weight binarization, i.e., restricts weightsin $\{ - \alpha , + \alpha \}$ . The authors propose to initialize BinaryBERTby equivalently splitting from a half-sized TernaryBERT toinherit the good performance of the ternary one. In addition,BiBERT [28] is a full binarization of BERT (i.e., 1-bit weight,embedding, and activation). The authors identify the severeperformance degradation of the full binary model comesfrom information degradation and optimization direction

mismatch. A Bi-Attention structure and a DirectionMatch-ing Distillation (DMD) scheme are proposed accordingly toretain most of the ability of the original BERT.

Some works enable an automatic balance betweenthe model performance degradation and quantization bit-width. Zhao et al. [29] leverages a Differentiable Neural Ar-chitecture Search approach to assign precision for parametersautomatically. In detail, the weights and the bit assignmentof weights are optimized alternatively under an objectivefunction that combines the cross entropy loss with thepenalty of model size. The optimization process aims toobtain a set of bit assignments for each group of parametersclose to optimal.

# 3.2.2 PTQ for Medium-Size Language Models

PTQ methods are carefully designed so they generally donot require extra finetuning or re-training to compensate forquantization errors. GOBO [33] quantizes the vast majorityof weights that comply with Gaussian distribution into 3 bitsusing non-uniform quantization (i.e., clustering) and saves afew outlier weights separately in FP32. I-BERT [34] designsinteger-only approximation methods for specific non-linearfunctions (e.g., GeLU, Softmax, LayerNorm) to enable end-to-end integer-only BERT inference without any floatingpoint calculation. Dai et al. [35] use finer granularity toreduce quantization error. In detail, the authors quantizeweights and activations into 4 bits using group-wise quan-tization (e.g., $1 6 \sim 6 4$ consecutive weights as a group). Acalibration set is used to determine the scaling factor foreach group.

Furthermore, it should be noted that the quantizationparameters obtained by elaborately tailored PTQ methodscan, in general, be a good initialization point for re-trainingin QAT methods.

3.2.3 Quantize Generative Medium-Size Language ModelsDespite the success of quantization approaches for BERT-like models mentioned above, attempts to quantize gener-ative language models (e.g., GPT, BART) was scarce beforethe emergence of generative LLMs [36]. The critical differ-ence is that quantization error accumulates in the token-by-token generation process, so quantizing generative languagemodels is generally a more complex problem.

According to Tao et al. [37], applying quantization meth-ods that are designed for BERT-like models directly togenerative language models is hindered by homogeneousword embedding and varied distribution of weights. Homo-geneous word embedding refers to the problem where theword embeddings of generative language models becomeless distinguishable from each other after quantization. Onthe other hand, varied distribution of weights means thatthe weights of the model are highly skewed with outliers. Totackle these challenges, the authors propose two solutions:token-level contrastive distillation and module-dependentdynamic scaling. DQ-BART [38] uses the QAT frameworkand a distillation training objective to distill and quantize asequence-to-sequence model, i.e., BART, jointly. DQ-BARTadopts the standard symmetric uniform quantization asshown in Equation (3) and sets the training objective asminimizing the differences of the output logits, attentions,and hidden states between the quantized and distilled

low-precision student model and the full precision teachermodel.

In this section, we only briefly cover the most impor-tant works done on medium-sized language models. For amore detailed summarization of quantization methods formedium-sized language models, we refer interested readersto [39], [40].

# 3.3 Post-Training Quantization for LLMs

The past few years have witnessed a remarkable surge inpost-training quantization methods (PTQ) for LLMs. This ispartly because PTQ doesn’t involve LLMs’ prohibitively ex-pensive re-training process, so it’s a more feasible directionfor most researchers.

Further, we roughly divide PTQ works for LLMs intotwo categories: weight-only quantization and weight + ac-tivation quantization. We’ll discuss works related to thesecategories respectively in the following parts.

# 3.3.1 Weight-Only Quantization.

In this part, we focus on the problem of only quantizing theweights (but not the activations) of LLMs. Generally, weight-only quantization belongs to simulated quantization meth-ods; the weights are only stored in low-bit datatype andneed to be dequantized before real computation. This meanssuch methods can decrease the overall size of LLMs anddecrease the time to move weights between memories butcannot enjoy the accelerated low-bit operation supported byspecific hardware.

While the previous subsections have discussed variousmethods that can be used to quantize medium-size languagemodels, LLMs present additional challenges due to theirunique characteristics. These challenges include:

1) LMs rely heavily on memory during the inferenceprocess, especially when the inference batch sizeis small [49]. This makes it crucial to minimizememory usage and optimize data transfer betweendifferent storage devices.

2) The activation patterns of LLMs are distinct, whichposes a challenge when applying quantizationmethods that work well for medium-sized languagemodels. Systematic outliers [43] are one such uniqueproperty of LLM activations that hinder the directapplication of such methods for weight-only quan-tization of LLMs.

Some works directly apply uniform, round-to-nearestquantization to LLMs with minor modifications [14], [21],[50]. ZeroQuant-V2 [21] quantize OPT and BLOOM. Itshows that using 16-bit activations and directly quantiz-ing weights of these models to 8-bit integers using row-wise symmetric quantization results in negligible perplexitydegradation, while 4-bit weight-only quantization witnessesa significant performance drop. To further push the limit oflow-bit quantization, ZeroQuant-V2 [21] propose the Low-Rank Compensation (LoRC) method, which approximatesthe error $\mathbf { E }$ between the original weight matrix W and thequantized weight matrix $\ddot { \mathbf { W } }$ using a storage-efficient low-rank matrix $\hat { \mathbf { E } }$ so that $\hat { \mathbf { W } } + \hat { \mathbf { E } }$ would be a better approx-imation of the original weight W. However, Zeng et al.


TABLE 1: Detailed category of several strong baseline quantization methods for LLMs. ✓means that a quantization methodbelongs to a specific category, $\times$ vice versus, and ◦ means a quantization method can be used in both circumstances. Formethods that work in different bit widths, we report the lowest effective bit width. For a detailed explanation of eachcategory, please refer to Subsection 3.1


<table><tr><td>Method</td><td>#Bits</td><td>Weight</td><td>Activation</td><td>Uniform</td><td>Symmetric</td><td>Static</td><td>Re-Training</td><td>Zero-shot</td><td>Integer-Only</td></tr><tr><td>AWQ [41]</td><td>3</td><td>group-size</td><td>-</td><td>✓</td><td>×</td><td>✓</td><td>×</td><td>×</td><td>×</td></tr><tr><td>OPTQ/GPTQ [42]</td><td>3</td><td>column-wise</td><td>-</td><td>✓</td><td>×</td><td>✓</td><td>×</td><td>×</td><td>×</td></tr><tr><td>LLM.int8() [43]</td><td>8</td><td>column-wise</td><td>row-wise</td><td>✓</td><td>○</td><td>×</td><td>×</td><td>✓</td><td>✓</td></tr><tr><td>ZeroQuant [44]</td><td>4</td><td>group-wise</td><td>tokenwise-wise</td><td>✓</td><td>✓</td><td>×</td><td>○</td><td>○</td><td>✓</td></tr><tr><td>SmoothQuant [45]</td><td>8</td><td>layer-wise</td><td>layer-wise</td><td>✓</td><td>✓</td><td>○</td><td>×</td><td>×</td><td>✓</td></tr><tr><td>LLM-QAT [46]</td><td>2</td><td>column-wise</td><td>tokenwise-wise</td><td>✓</td><td>✓</td><td>×</td><td>✓</td><td>×</td><td>×</td></tr><tr><td>INT2.1 [47]</td><td>2</td><td>column-wise</td><td>-</td><td>✓</td><td>×</td><td>✓</td><td>✓</td><td>×</td><td>×</td></tr><tr><td>QLoRA [48]</td><td>4</td><td>column-wise</td><td>-</td><td>×</td><td>✓</td><td>✓</td><td>✓</td><td>×</td><td>×</td></tr></table>

[14] found that GLM-130B can be directly quantized into4-bit weights using row-wise symmetric quantization withnegligible performance degradation, which is evaluated byzero-shot accuracy on the LAMBADA dataset. The authorsascribe the appealing 4-bit quantization property of GLM-130B to its weight distributions being well-shaped andnot skewed compared to GPT-style models like OPT andBLOOM.

Another line of research considers non-uniform methodsin weight-only quantization of LLMs. The critical insightlies in the fact that the weight distribution of LLMs aftertraining is non-uniform, so it makes sense to let interval$[ \Delta _ { i } , \Delta _ { i + 1 } )$ in Equation (4) also be non-uniform to push thequantization even to lower bit-width. LUT-GEMM [51] (alsoknown as nuQmm) extends a non-uniform quantizationmethod, binary-coding quantization (BCQ) [52], which fac-torizes the full-precision parameters into binary parametersand a separate set of scaling factors. The authors add a biasterm to conventional BCQ methods to increase the represen-tational capacity and use group-wise quantization to enablea tradeoff between the compression ratio and model per-formance. SqueezeLLM [49] verifies that the LLM inferenceis memory-bound with extremely low arithmetic intensityrelative to other neural networks. Besides, SqueezeLLMadopts sensitivity-based k-means centroids as the quantizedweight values for non-uniform quantization (see $X _ { i }$ inEquation (5)). The sensitivity-based k-means method ap-proximates the Hessian matrix of weights as the sensitivity,highlighting the importance of minimizing perturbations forweights with large Hessian values. SqueezeLLM has betterperplexity than standard uniform quantization methodswhile achieving around $2 \times$ speedup compared to the FP16baseline. Dettmers et al. [48] propose a new NormalFormat(NF) datatype, which can also be viewed as non-uniformquantization. The NF datatype builds on Quantile Quan-tization [53], an information-theoretically optimal data typethat ensures each quantization interval has an equal numberof values assigned from the input tensor. The authors utilizethat pre-trained neural network weights usually have azero-centered normal distribution with standard deviation$\sigma _ { \iota }$ , thus can be normalized to the standard normal distri-bution $N ( 0 , 1 )$ by scaling σ. $k$ -bit NormalFormat use $k \mathrm { . }$ -bit Quantile Quantization on standard normal distribution

$N ( 0 , 1 )$ to find its quantized values $Q _ { i }$ (See Equation (5)for definition of $Q _ { i } { \mathrm { . } }$ ). In practice, weights to be quantizedare rescaled to range $[ - 1 , 1 ]$ and then round to nearestquantized values $X _ { i }$ i.e., round-to-nearest (RTN) methods.

Above are zero-shot methods that only consider the min-imize the difference between the original weight matrix Wand the quantized weight matrix $Q ( { \dot { \mathbf { W } } } )$ , i.e., to minimize thequantization error of weight matrix $\mathrm { a r g m i n } _ { \hat { \mathbf { W } } } | | \mathbf { W } - Q ( \mathbf { W } ) | |$ .However, considering the high non-linearity of neural net-works, a small distance in weight space doesn’t necessarilymean a small difference between the output of the originaland quantized models. Thus, if given a small set of typicalexamples $C ,$ called calibration set, there are some one-shotmethods [41], [42], [54], [55] consider to optimize the dif-ference between the output activations of the original andquantized layers:

$$
\operatorname {a r g m i n} _ {\hat {\mathbf {W}}} \left\| \mathbf {X} \mathbf {W} - \mathbf {X} Q (\mathbf {W}) \right\| \quad \text {f o r} \mathbf {X} \in C \tag {7}
$$

A typical work of one-shot methods for weight-onlyLLM quantization is GPTQ (also known as OPTQ) [42],which is built on an adaptive rounding method called Opti-mal Brain Quantization (OBQ) [56]. OBQ handles each rowof the weight matrix independently, quantizing one weightat a time while updating all not-yet-quantized weights tocompensate for the error incurred by quantizing a singleweight. However, OBQ is not explicitly designed for LLMsand can be slow and inaccurate in practice. To fix theseproblems, GPTQ quantizes weights of all rows in parallelto improve efficiency, uses lazy batch updates to achieve ahigher compute-to-memory ratio in the quantization pro-cess, and uses Cholesky reformulation to help numericalstability. GPTQ can quantize OPT-175B or BLOOM-176B inaround 4 hours on a single NVIDIA A100 GPU with thesemodifications. Further, GPTQ can provide reasonable accu-racy under extreme quantization where weights are quan-tized to 2-bit or lower. QuIP [55] defines a family of adaptiverounding methods for optimizing the Equation (7) anddefines the optimal method within the pre-defined family ofmethods, called LDLQ. LDLQ uses the LDL decompositionof the second-moment matrix of vectors in the calibrationset to find the optimal way to update the not-yet-quantizedweights. The authors show that GPTQ is a particular caseof LDLQ. Further, QuIP proposes incoherence processing

that can transform the weight matrix into a more suitableform for quantization. Combining LDLQ and incoherenceprocessing, QuIP is the first LLM quantization method thathas viable results on 2-bit weight-only quantization. AWQ[41] shows that preserving only $0 . 1 \%$ channels correspond-ing to significant activation in FP16 and quantizing the restof the weight matrix can contribute to much better modelperformance, meaning weights are not equally important.Further, AWQ intends to reduce the quantization errorfor the essential weights without using mixed-precision,which is hard to implement efficiently. This is achieved byactivation-aware scaling, which automatically finds a per-(input) channel scaling ratio s with an objective similarto Equation (7) i.e., $\mathrm { a r g m i n _ { s } | | ( s ^ { - 1 } \cdot X ) } Q ( W \cdot s ) \ - \ \mathbf { X } \mathbf { W } | |$such that the salient weights with high scaling factor can bebetter represented. In contrast, non-salient weights will notbe neglected. OWQ [57] is an advancement over OPTQ thatsignificantly enhances the quality of the quantized model. Ituses a mixed-precision quantization scheme, which applieshigher precision to the weights susceptible to quantizationcaused by activation outliers. The sensitive weights areidentified using a sensitivity-based method similar to [49].

There are also some studies focusing on rounding cri-teria in quantization. SignRound [58] suggests that as thebit-width of quantization decreases, the quantization gridbroadens, thus emphasizing the importance of up and downrounding. It extends previous work [59] to learn weightrounding with signed gradient descent and can achievegood results within 400 optimization steps.

# 3.3.2 Weight $^ +$ Activation Quantization.

Quantizing both weights and activations of LLMs is a non-trivial problem for several reasons. First, the range andstatistics of activation are unknown until actual runtime.Second, quantizing weights and activations enables efficientlow-bit datatype operations on specific hardware. Third,systematic outliers appearing in the LLM activations arevital to the model performance, and shouldn’t be clipped inquantization. While the first two reasons apply to all mod-els, the third reason is unique to LLMs and differentiatesmethods for LLMs from methods for previous models.

Similar to its weight-only counterpart, weight $^ +$ activa-tion quantization can also use basic uniform quantizationmethods [21], [43], [44], [63] but with a special notificationof outliers in activations. LLM.int8() [43] emphasizes theemergence of extreme outliers in LLM’s activations as themodel size scales up. The authors show that these out-liers are highly systematic. Given input activation $X _ { f 1 6 } \in$$\mathbb { R } ^ { N \times D _ { \mathrm { i n } } }$ to a linear layer, outliers occur systematically foralmost all $N$ tokens in a sequence. Still, they are limitedto specific feature/hidden dimensions $\hat { d } \in \{ \stackrel { . } { 1 } , 2 , . . . , D _ { \mathrm { i n } } \}$ .LLM.int8() thus propose to separate the outlier featuredimensions $O = \{ \hat { \hat { d } } | \hat { d } \stackrel { \bullet } { \in } \mathbb { Z } , 1 \leq \hat { \hat { d } } \leq D _ { \mathrm { i n } } \}$ which containsall feature dimensions $\hat { d }$ that have at least one activationoutlier with a magnitude more significant than the threshold$\alpha$ . The outlier dimensions are preserved in high-precisiondatatypes (e.g., FP16) while average values are quantizedusing symmetric uniform quantization into low-precisiondatatypes (e.g., INT8). With Einstein’s notation, the matrix

multiplication thus becomes:

$$
\mathbf {X} _ {f 1 6} \mathbf {W} _ {f 1 6} \approx \sum_ {\hat {d} \in O} \mathbf {X} _ {f 1 6} ^ {\hat {d}} \mathbf {W} _ {f 1 6} ^ {\hat {d}} + \mathbf {S} _ {f 1 6} \cdot \sum_ {d \notin O} \mathbf {X} _ {i 8} ^ {d} \mathbf {W} _ {i 8} ^ {d} \tag {8}
$$

where $\mathbf { S } _ { f 1 6 }$ is the dequantization factor. The number ofoutlier dimensions $| O |$ is quite small, so this decompo-sition would only consume more than $0 . 1 \%$ additionalmemory typically. Instead of separating outliers into anadditional matrix, RPTQ [63] proposes to cluster and re-order the dimensions of activation $\pmb { \chi } \in \mathbb { R } ^ { N \times D _ { \mathrm { i n } } }$ based onthe minimum and maximum of the dimension $i ,$ denotedas $( { \pmb X } _ { m i n , i } , { \pmb X } _ { m a x , i } )$ . The idea is to group dimensions withoutliers into the same cluster and perform cluster-wisequantization. It should be noted that the statistics of eachactivation dimension are measured on a calibration set sothat the clustering can be done before inference to findthe new order of dimensions. To further reduce latency,RPTQ fuses the reorder operation into other operations: 1)Combine with the LayerNorm operation to avoid additionaldata movement and adjust. 2) Reorder columns of weight Wto reorder the dimensions of the output $\mathbf { Y } = \mathbf { X } \mathbf { W }$ .

Recently, low-bit floating-point formats (e.g., FP4, FP8)have emerged as promising alternatives for LLM quanti-zation [64], [65], [66]. FP8 format has garnered supportfrom leading hardware vendors like NVIDIA despite itspotentially higher hardware costs. Intuitively, low-bit FPformats can be viewed as a particular case of non-uniformquantization, offering a typically more extensive data rangeand higher precision for small values but lower precision forlarge ones. Such characteristics of the FP format help solveoutlier problems in activations. MoFQ (Mixture-of-FormatsQuantization) [64], and ZeroQuant-FP [65] both show thatFP8 quantization is consistently better than INT8 when itcomes to activation quantization. MoFQ further provides analgorithm to determine the optimal data format from somecandidates (INT4, INT8, FP4, FP8) for each layer based ontensor error, layer output error, or model output error. Also,MoFQ reallocates special NaN (Not a Number) and Inf (In-finity) values in standard FP formats to normalized numbersto enhance and let the FP format represent more values,which is especially important for low-bit formats like FP4.ZeroQuant-FP quantizes both weight and activation into FPformat. For cases using FP4 weights and FP8 activations,ZeroQuant-FP proposes a bit-shifting method to cast FP4 toFP8 to improve inference efficiency efficiently.

Another promising way is to suppress outliers that ap-pear in the activation dimensions [45], [67], [68], [69], [70].The general idea is that we can scale down the outlierdimensions $i$ in activations by factor $\mathbf { s } _ { i }$ and scale up thecorresponding dimension in the weight matrix by factor ${ \bf s } _ { i }$without changing the output of the layer:

$$
\mathbf {Y} = \mathbf {X} \mathbf {W} = (\hat {\mathbf {X}} \operatorname {d i a g} (\mathbf {s})) \cdot (\operatorname {d i a g} (\mathbf {s}) ^ {- 1} \hat {\mathbf {W}}) = \hat {\mathbf {X}} \hat {\mathbf {W}} \tag {9}
$$

so that the activation $\hat { \mathbf { x } }$ after scaling would be quantization-friendly. SmoothQuant [45] computes the per-dimensionscaling factor using:

$$
\mathbf {s} _ {i} = \max  \left(\left| \mathbf {X} _ {i} \right|\right) ^ {\alpha} / \max  \left(\left| \mathbf {W} _ {j} \right|\right) ^ {1 - \alpha} \tag {10}
$$

where $\alpha$ is a hyperparameter to control how much difficultywill be migrated from activation to weights. Outlier Sup-pression [67] discovers that $\gamma$ in LayerNorm acts as a sinful

TABLE 2: The following table shows the perplexity of various strong baseline quantization methods for LLMs on Wikitext-2 [60] and C4 [61]. Please note that the intention is not to compare the perplexity after quantization directly, as differentquantization methods may perform best on different models with different scales. This table only serves as a roughcomparison of the effects of different quantization methods. We strongly recommend readers refer to the original papersfor detailed results in different settings. The reported results are derived from the original papers, except for QLoRA, whoseresult is derived from LoftQ [62].

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Method</td><td rowspan="2">#Bits Weights</td><td rowspan="2">#Bits Activations</td><td rowspan="2">Model</td><td colspan="2">Perplexity (↓)</td><td rowspan="2">Speedup</td></tr><tr><td>FP16 Model</td><td>Quantized Model</td></tr><tr><td rowspan="7">Wikitext-2</td><td>AWQ [41]</td><td>3</td><td>16</td><td>OPT-66B</td><td>10.09</td><td>10.46</td><td>1.85×</td></tr><tr><td>OPTQ/GPTQ [42]</td><td>3</td><td>16</td><td>OPT-175B</td><td>8.34</td><td>8.68</td><td>3.24×</td></tr><tr><td>ZeroQuant [21]</td><td>4</td><td>8</td><td>BLOOM-176B</td><td>8.11</td><td>8.33</td><td>-</td></tr><tr><td>SmoothQuant [45]</td><td>8</td><td>8</td><td>LLaMA-65B</td><td>6.17</td><td>6.20</td><td>1.56×</td></tr><tr><td>LLM-QAT [46]</td><td>4</td><td>8</td><td>LLaMA-30B</td><td>7.00</td><td>7.50</td><td>-</td></tr><tr><td>INT2.1 [47]</td><td>2</td><td>16</td><td>LLaMA-7B</td><td>5.08</td><td>8.74</td><td>-</td></tr><tr><td>QLoRA [48]</td><td>3</td><td>16</td><td>LLaMA-13B</td><td>5.12</td><td>5.22</td><td>-</td></tr><tr><td rowspan="5">C4</td><td>OPTQ/GPTQ [42]</td><td>3</td><td>16</td><td>OPT-175B</td><td>10.13</td><td>10.67</td><td>3.24×</td></tr><tr><td>LLM.int8() [43]</td><td>8</td><td>8</td><td>OPT-13B</td><td>12.45</td><td>12.45</td><td>1.22×</td></tr><tr><td>ZeroQuant [21]</td><td>4</td><td>8</td><td>BLOOM-176B</td><td>10.97</td><td>11.22</td><td>-</td></tr><tr><td>LLM-QAT [46]</td><td>4</td><td>8</td><td>LLaMA-30B</td><td>6.00</td><td>6.90</td><td>-</td></tr><tr><td>INT2.1 [47]</td><td>2</td><td>16</td><td>LLaMA-7B</td><td>7.52</td><td>12.52</td><td>-</td></tr></table>

amplifier for the outliers. Hence, it proposes the GammaMigration method, which uses $\gamma ^ { - 1 }$ in the previous layer asthe scaling factor s. Outlier Suppression $^ +$ [68] extends themethod through introducing additional shifting factor z:

$$
\begin{array}{l} \mathbf {Y} = \mathbf {X} \mathbf {W} = (\hat {\mathbf {X}} \operatorname {d i a g} (\mathbf {s}) + \mathbf {z}) \cdot (\operatorname {d i a g} (\mathbf {s}) ^ {- 1} \hat {\mathbf {W}}) \tag {11} \\ = \hat {\mathbf {X}} \hat {\mathbf {W}} + \mathbf {z} \mathbf {W} = \hat {\mathbf {X}} \hat {\mathbf {W}} + \hat {\mathbf {b}}. \\ \end{array}
$$

The per-dimension shifting factor is computed as $\begin{array} { r l } { \mathbf { z } _ { i } } & { { } = } \end{array}$$( \operatorname* { m a x } ( \mathbf { X } _ { i } ) + \operatorname* { m i n } ( \mathbf { X } _ { i } ) ) / 2 ,$ which helps remove the asym-metry in the activation $\hat { \textbf { \textit { X } } } = \mathbf { \kappa } ( \mathbf { X } - \mathbf { \dot { \omega } } \mathbf { z } ) \mathrm { d i a g } ( \mathbf { s } )$ . FPTQ [69]proposes a new offline logarithmic activation equalization(LAE) method that moderates activation distributions in anon-linear fashion, each channel of the scaling factor s iscomputed as:

$$
\mathbf {s} _ {i} = \max  \left(\left| \mathbf {X} _ {i} \right|\right) / \log_ {2} (2 + \max  \left(\left| \mathbf {X} _ {i} \right|\right)) \tag {12}
$$

While the above methods use hand-craft quantizationparameters such as scaling factor s, Outlier Suppression+in contrast proposes to find optimal scaling factor s byoptimizing the following objective using a calibration set:

$$
\min  _ {\mathbf {s}} \mathbb {E} [ \| \mathbf {X} \mathbf {W} - (Q (\hat {\mathbf {X}}) Q (\hat {\mathbf {W}}) + \hat {\mathbf {b}}) \| _ {2} ^ {2} ]. \tag {13}
$$

Further, OmniQuant [70] proposes learning the clippingrange $[ \alpha , \beta ]$ to modulate the extreme values of weights.QLLM [71] proposes a unique approach to handling outliersin activations. The technique involves an adaptive channelreassembly process that splits the outlier channels into mul-tiple sub-channels. This ensures a more even distributionof activation magnitudes. The process also merges similarchannels, maintaining the original channel number for effi-ciency purposes.

There are also some recent studies employing methodsthat do not fall into any of the previous paragraphs, so webriefly cover them here for completeness. REx [72] quantizesthe quantization error, i.e., $\mathbf { W } \mathrm { ~ - ~ } Q ( \mathbf { W } ) .$ , so that there is a

smaller quantization error between the original value andthe dequantized value, which trades efficiency for highermodel performance. OilVe [73] employs the outlier-victimpair (OVP) mechanism, which prunes some quantized low-precision normal values to make extra space for the high-precision outliers.

As a summary of PTQ methods for LLMs quantiza-tion, we briefly compare and contrast the weight-onlyand weight+activation quantization methods here. On theone hand, weight-only quantization methods can push thequantization limit to lower bit-widths (even to 3 bits or 2bits), which significantly reduces the memory size of devicesrequired by LLMs. This is because model weights use mostof the memory. On the other hand, weight+activation quan-tization can take advantage of the additional speedup thatcomes with efficient low-bit arithmetic supported by specifichardware and doesn’t introduce additional dequantizationoverhead during inference. However, these methods oftenrequire more bit-width $\sim 8$ bits) to store the weights andactivations. Both weight-only and weight+activation quan-tization methods have their strengths and weaknesses, andthey are both active research directions with great potentialand demand.

# 3.4 Quantization-Aware Training for LLMs

Quantization-Aware Training (QAT) is the method of re-training a quantized model to recover from the performancedegradation caused by the quantization. As is illustrated inthe previous sections, QAT for models before LLMs (e.g.,CNN, medium-size LM) has achieved impressive success.However, such methods often involve full-parameter re-training of the entire model, which is too expensive forLLMs, so there are also some attempts to combine quan-tization with parameter-efficient training methods to signif-icantly lower the cost of QAT on LLMs.

As a result, we divide current QAT methods on LLMsinto two categories: full-paramter re-training and parameter-efficient re-training. We’ll discuss works in these two cate-gories respectively in the following parts.

# 3.4.1 Full-Parameter Re-Training.

The primary concern of using the QAT framework in LLMsis re-training them on a smaller dataset without hurtingtheir emergent abilities, such as in-context learning. Currentmethods often combine QAT and distillation to preservethese abilities of the original (teacher) model [74], [75].

LLM-QAT [46] directly applies the basic QAT framework[24] to LLMs. To cope with the problem, LLM-QAT proposesto use a data-free distillation method, where the data isgenerated using the original LLM, and the quantized LLM istrained to match the output distribution of the original LLMon the generated data. Also, LLM-QAT enables quantizationand QAT of key-value caches, which takes up large memoryin the long sentence generation process.

To alleviate the unbearable cost of re-training full LLM,ZeroQuant [21], [44], [65] proposes a layer-to-layer knowl-edge distillation method. The method uses the original LLMas a teacher and processes the weights of LLM in a layer-by-layer order. Assume the LLM has $N$ layers $L _ { 1 } , L _ { 2 } , \dots , L _ { N }$and has some input dataset $\mathbf { \boldsymbol { x } }$ and the quantized ver-sion of $L _ { k }$ is $Q ( L _ { K } )$ . After QAT and distillation of layers$L _ { 1 } , L _ { 2 } , \dots , L _ { k - 1 } ,$ we use the following loss to update $L _ { k }$ :

$$
\mathcal {L} _ {k} = \operatorname {M S E} \left(L _ {k} \cdot L _ {k - 1} \dots L _ {1} (\mathbf {X}) - \hat {L _ {k}} \cdot L _ {k - 1} \dots L _ {1} (\mathbf {X})\right) \tag {14}
$$

# 3.4.2 Parameter-Efficient Re-Training.

There are a bunch of works using parameter-efficient meth-ods (e.g., LoRA, Adapter, Prompt Tuning) to finetune theLLMs, which will be discussed in Appendix A. In this sec-tion, we discuss some methods that use parameter-efficientmethods in the re-training process of QAT.

Typical works [47], [48], [62], [71], [76], [77], [78] adoptLow-Rank Adaption (LoRA) to re-train quantized LLMsin a relatively acceptable compute budget. QLoRA [48]quantize the weight of LLMs into 4-bit NormalFormat andsubsequently adopt LoRA with 16-bit BrainFloat to finetunethe quantized model on downstream tasks with cross en-tropy loss. It further introduces a technique named doublequantization, which quantizes the quantization parametersto compress further the model’s size in the trade of com-putation speed. Combining all these techniques, QLoRAenables finetuning a 65B LLM on a GPU with 30G memoryefficiently. Following QLoRA, QA-LoRA [78] proposes tointegrate group-wise quantization into QLoRA. The authorssuggest that quantization parameters in QLoRA are muchless than LoRA parameters, resulting in an imbalance be-tween quantization and low-rank adaptation, while group-wise operations can alleviate the problem by increasing thenumber of parameters of quantization and decreasing thatof adaptation. Besides, LoftQ [62] finds the zero initializa-tion of LoRA matrices in QLoRA inefficient for downstreamtasks. Instead, LoftQ proposes to initialize the LoRA ma-trices with the singular value decomposition (SVD) of thedifference between the original and quantized weights, i.e.,$\mathbf { W } { - } Q ( \mathbf { W } )$ . LoftQ alternates between quantization and SVDto obtain a better approximation of the original weights.

LACos-BLOOM [76] quantizes the model weights using 8-bit block-wise uniform quantization. The quantized modelis then finetuned using a scalable LoRA and 8-bit Adamoptimizer. INT2.1 [47] utilized GPTQ to quantize LLMs intoINT2 and found that the behavior of the quantized modeldeviates significantly from the original full-precision coun-terpart. INT2.1 integrates additional trainable parameters(LoRA matrices) into the model and solely updates theLoRA matrices with takes up of only $5 \%$ of total parameters.The training objective combines a scaled Kullback-Leiblerdivergence from the full precision model to the quantizedone and the cross entropy loss to encourage accurate nexttoken prediction. Their experiment indicates that an INT2Large Language Model (LLM) finetuned with LoRA cangenerate linguistically coherent English text and exhibitadherence to prescribed instructions.

Other works [79], [80] freeze the quantization indicesand solely finetune quantization parameters (e.g., scalingfactor $S$ in uniform quantization and quantization level$\Delta _ { i }$ in non-uniform quantization). AlphaTuning [79] worksby employing binary-coding quantization [52]. During theadaptation phase, the binary values are frozen for all tasks,while the scaling factors are fine-tuned for the downstreamtask. PEQA [80] quantizes each fully connected layer of LMsinto a matrix of low-bit integers and a scalar vector usinguniform quantization. Subsequently, finetuning occurs onthe scalar vector for each downstream task.

Works also combine quantization with adapters [81] andprompt tuning [82].

# 3.5 Other Topics for LLM Quantization

Some quantization-related works can not be categorizedinto PTQ or QAT; we discuss such works in this section.

One important topic is co-designing efficient kernelsalong with quantization algorithms [83], [84], designinghardware-friendly quantization methods [85], [86] and in-tegrating quantization methods in real-world applications[87], [88], [89], [90], [91]. LUT-GEMM [51] is an efficient ker-nel designed for an extended version of BCQ methods [52],which can represent both uniform and non-uniform quan-tization. Since weights are characterized by a binary vectorand scale factors in BCQ, LUT-GEMM can pre-compute andstore all possible combinations of full-precision activationsand binary patterns in a lookup table (LUT) to avoid re-peated computation and remove dequantization of weights,which accelerates the latency of OPT-175B model with 3-bitquantization by $2 . 1 \times$ compared to conv entional simulatedquantization. Many uniform [21], [43], [44], [73] and non-uniform quantization methods [49] discussed in the abovesections also design special kernels to reduce the overalllatency.

Other meaningful works study the intrinsic character-istics of LLM quantizations [92], [93], [94]. For example,Dettmers and Zettlemoyer [93] run extensive experimentswith 16-bit activations and $k$ -bit weights $( 3 ~ \leq ~ k ~ \leq ~ 8 )$ at scales of 19M to 176B parameters across LLM familiesBLOOM, OPT, NeoX/Pythia and GPT-2. The authors focuson the tradeoff between zero-shot ability and total modelbits and show that 4-bit precision is almost universallyoptimal for the tradeoff across different LLM classes and

quantization methods. Liu et al. [94] aim to investigatethe impact of quantization on emergent abilities, which areessential characteristics that distinguish LLMs from smalllanguage models. Their empirical experiments show thatemergent abilities still exist in 4-bit quantization models,while 2-bit models encounter severe performance degra-dation on the test of these abilities. The authors conductfurther detailed experiments on enhancing the performanceof extremely low-bit models.

Some works [67], [95], [96] also focus on studying thereasons behind the emergence of systematic outliers inLLMs and looking for ways to suppress the outliers from thesource. Quantizable Transformer [95] ascribes the outliers inactivations to the behavior of attention heads that try notto update residual. The authors designed clipped softmaxand gated attention accordingly to grant the model theability to produce minimal magnitude (or even exact zeros)output of attention function without having outliers. Outliersuppression [67], however, treats $\gamma$ in LayerNorm as thesinful amplifier of outliers. There is still no consensus onthe source of activation outliers. However, Ahmadian et al.[96] find that outlier dimensions may not be an inherentproduct of scale as is thought in previous works [43], butrather sensitive to the optimization conditions (e.g., dropoutrate, weight decay, datatype) present during pre-training.

# 4 PRUNING

As a conventional technique employed for the compressionand acceleration of neural networks, pruning eliminatesnon-essential weights or structures from models, while pre-serving the performance of the networks at a level nearlyequivalent to their original state. Although pruning hasshown remarkable results in CNNs [97], its effectiveness isless robust for LLMs when compared to other compressiontechniques such as quantization and distillation. The reasonwhy pruning becomes less effective comes from the fine-tuning process. The high cost of fine-tuning due to thelarge number of model parameters makes it more difficultto achieve the full effect of pruning. Nevertheless, pruningis a crucial technique for compressing models, necessitatingfurther exploration to enhance and refine its effectiveness inyielding improved results in LLMs.

In the following section, we will provide an overviewof pruning methods and basic concepts in Section 4.1. Sub-sequently, in Section 4.2, we will expound upon pruningtechniques tailored for medium-size language models (i.e.,models with parameters in billions), given their structuralsimilarities with LLMs. Section 4.3 will delve into a detailedexploration of pruning methodologies specifically designedfor LLMs. Finally, in Section 4.4, we will introduce someauxiliary techniques that are not pruning methods but asso-ciated with pruning to improve LLM pruning results, andthen discuss the challenges for future advancements in thefield of LLM pruning.

# 4.1 Basic Concepts

Numerous classification criteria exist for pruning. Never-theless, the most significant things among them are twofundamental problems: what to prune and how to prune.

The answers to these two problems correspond respectivelyto the pruning unit and metric. We will introduce these twofundamental concepts and some other basic ones.

1) Pruning Unit. The first fundamental problem withpruning is what kind of elements should be pruned. Thepruning units refer to the minimal pruning elements in thepruning process, encompassing elements such as weights,neurons, attention heads, layers, and etc. Based on pruningunits, pruning methods can be broadly categorized into un-structured pruning and structured pruning. In unstructuredpruning, the pruning units focus on individual weights. Theweights to be pruned are zeroed out. Whereas in structuredpruning, the pruning units encompass broader networkstructures, such as neurons, attention heads, and layers. Thestructures to be pruned are removed from the networks.

Unstructured pruning tends to get a higher sparsity ratioand maintain better performance as it is not limited tothe network structure and can prune individual weights.However, the irregularly sparse patterns of weight matri-ces, stemming from the non-systematically occurring zerovalues, exhibit computational efficiency nearly equivalentto dense matrices. Consequently, achieving significant gainsin inference speedup is infrequent in unstructured pruning.

Structured pruning makes it easy to achieve inferencespeedup as it prunes network structures (e.g., attentionheads, feed-forward network (FFN) neurons, and hiddendimensions). Yet inevitably integrated structure deletionsmay cause the performance descent of the model. To avoidmodel collapse, the sparsity ratios of structured prunedmodels are lower than unstructured ones.

Formally, a binary mask z usually covers the pruningunit during the pruning process and is multiplied into themodel after pruned. For unstructured pruning, the pruningprocess can be defined as a constrained optimization prob-lem:

$$
\min  _ {\mathbf {w}, \mathbf {z}} \mathcal {L} (\mathbf {w} \odot \mathbf {z}; \mathcal {D}) = \min  _ {\mathbf {w}, \mathbf {z}} \frac {1}{N} \sum_ {i = 1} ^ {N} \ell (\mathbf {w} \odot \mathbf {z}; (x _ {i}, y _ {i})), \tag {15}
$$

$$
s. t. \quad \| \mathbf {z} \| _ {0} \leq t,
$$

where $\odot$ corresponds to the element-wise product, $\mathbf { w } =$$\left\{ w _ { 1 } , w _ { 2 } , . . . , w _ { M } \right\}$ is the network weights, $\mathcal { D }$ is a datasetcomposed of $N$ input $x _ { i }$ and output $y _ { i }$ pairs, and $t$ isthe target non-sparsity ratio (i.e., one minus sparsity ratio).Similarly, the pruning process for structured pruning is asfollows:

$$
\min  _ {\mathbf {w}, \mathbf {z}} \mathcal {L} (\mathbf {s} \odot \mathbf {z}; \mathcal {D}) = \min  _ {\mathbf {w}, \mathbf {z}} \frac {1}{N} \sum_ {i = 1} ^ {N} \ell (\mathbf {s} \odot \mathbf {z}; (x _ {i}, y _ {i})), \tag {16}
$$

$$
s. t. \quad f (\mathbf {z}; \mathbf {s}) \leq t,
$$

where $\textbf { s } = ~ \{ s _ { 1 } , s _ { 2 } , . . . , s _ { K } \}$ is the pruning structures com-posed of w, and $f ( \cdot )$ is the function to compute non-sparsityratio according to the binary masks and structures.

2) Pruning Metric. The second fundamental problemwith pruning is how to determine whether an elementis essential and should be pruned or not. Pruning metricis the answer to this problem. The pruning metric is thecriterion to identify the importance of the pruning units. Itcan be roughly divided into three parts: magnitude-based,loss-based (i.e., considering the first-order and second-order

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-11/0a775777-6cdd-4dbe-a304-295f64b9a1ad/f93545d95881db3232d6f0e63070439c419b72b3ef03e867f0e4efba66d8ea58.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-11/0a775777-6cdd-4dbe-a304-295f64b9a1ad/9643dc01ddd53f2b1d640fcea7d50d4873530e5a1970fcb3189c75e22421d001.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-11/0a775777-6cdd-4dbe-a304-295f64b9a1ad/27764fed42caa63ff8b2d297d0c6dd7dbb7c8f6c2f0d391f04ffb7817f6f995b.jpg)



Fig. 3: Three classes of static pruning methods. (a) Pre-training pruning; (b) During-training pruning; (c) Post-training pruning.


derivative information of the weighs belonging to the prun-ing units), and regularization.

The magnitude-based pruning methods use the magni-tudes (i.e., absolute values) of weights and activation valuesas a part of the pruning metrics. The fundamental principleunderlying this class of methods is that the magnitude ofweight or the activation value from the pruning unit intu-itively reflects its importance. The magnitude of the weightalone can serve as a pruning metric, constituting a well-known foundational pruning method known as MagnitudePruning [98]. Magnitude Pruning is the vanilla magnitude-based pruning method. In this method, a threshold is set tozero out weights with smaller magnitude and the thresholdtypically is derived from sparsity ratio. Despite the defini-tion of importance score being quite heuristic, MagnitudePruning demonstrates efficacy across various models.

In addition to the intuitive magnitude-based metric,another more sophisticated kind of metric is the loss-basedmetric. The loss-based metric is designed to attribute theimportance of a pruning unit to its impact on loss. If the lossincreases significantly after pruning an element, it indicatesthat the element should not be pruned. More precisely,following the pruning of an element, the greater the increasein loss, the more crucial the importance of that elementbecomes. However, examining the loss after pruning indi-vidual elements one by one is resource- and time-intensive.In contrast, employing the Taylor expansion provides amore convenient expeditious method for elucidating theloss alteration. The alteration in loss after the pruning canbe quantified using a Taylor expansion, incorporating thefirst-order or second-order derivatives of the pruning unitswith respect to the loss and higher-order ones, which areusually ignored. In comparison to the resource- and time-intensive approach of evaluating loss after pruning eachelement individually, the computation of the first-order andsecond-order derivatives emerges as a more efficient andtime-saving alternative.

Besides, regularization methods encompass L0, $L _ { \mathrm { 1 } } ,$ and$L _ { 2 }$ regularization. While $L _ { 1 }$ regularization is known forinducing sparsity in weights, $L _ { 0 }$ regularization is a morecommonly employed regularization approach in the contextof pruning.

3) Dynamic/Static Pruning. To enhance adaptabilityto diverse inputs, a kind of pruning method, referred toas dynamic pruning, constructs the network in a mannercontingent upon the specific input characteristics. We willthese later in Section 7. In contrast, static pruning methodsprune the model at training time and fix the architecture

after pruning, thus different inputs share the same prunednetwork. Static pruning methods can be classified as pre-training pruning, during-training pruning and post-trainingpruning according to the pruning period, as shown in Fig. 3.

Pre-training pruning: prunes the initialized networkfirst and then trains the sparse network.

During-training pruning: trains and prunes the densenetwork at the same time, where regularizationmethods are representative.

Post-training pruning: is the most popular type ofpruning pipeline, prunes the trained dense networkto get a sparse network, where they usually followa training, pruning, and fine-tuning paradigm as wementioned before.

4) Iterative/One-shot Pruning. As pruning damagesmodel performance inevitably, a popular paradigm of prun-ing pipeline consists of three steps: training, pruning, andfine-tuning, as shown in Fig. 3 (b), (c). The initial stepinvolves training the network to ascertain the importanceof individual pruning units. Subsequently, the second stepentails the removal of non-essential pruning units throughpruning, and the third step focuses on fine-tuning to recoverthe performance of the model post-pruning.

Given the potential for the fine-tuning process to renderinitially zero-valued weights as non-zero, the final two stepsare subject to iterative repetition until the targeted sparsityratio is achieved. This iterative design underscores that eachpruning step is succeeded by a fine-tuning step, therebyfacilitating the preservation of the model’s performance.These methods containing iterative pruning and fine-tuningrounds are classified as iterative pruning.

However, as the model parameters get huger, the itera-tive pruning and fine-tuning process is expensive and time-consuming. Thus more pruning methods tend to prune thenetwork only once to the target sparsity ratio, discarding theiterative pruning and fine-tuning rounds. These methodsare classified as one-shot pruning.

5) Global/Local Pruning. The early pruning approachescompare all the pruning units to identify and eliminatethose less essential. Given that the comparison scope inthese methods encompasses the entire network, they arecategorized as global pruning approaches. However, globalpruning permits distinct sparsity ratios for individual localregions. It might result in excessive pruning of a specificregion (e.g., a layer, a column), and exert a notable influenceon the overall performance of the model. The resolution ofthis issue lies in the application of local pruning method-ologies. Local pruning imposes constraints on the sparsityof each region, thereby ensuring that the sparsity ratioswithin each region do not reach excessively low thresholds,consequently mitigating the risk of model collapse.

6) Data-driven/Data-free Pruning. The categorization ofpruning methods into data-driven and data-free modalitiesdistinguishes the reliance on data for pruning decisions.Specifically, data-driven pruning methods, exemplified by themajority of pruning techniques, derive pruning decisionsfrom available data. Conversely, data-free pruning methods,such as Magnitude Pruning [98], execute network pruningindependent of data input. In general, data-driven pruningmethods tend to exhibit superior performance, given their

dependence on data-driven insights, while data-free prun-ing methods are less effective but data-independent.

7) Upstream/Downstream Pruning. The training of lan-guage models involves two main stages—pre-training andfine-tuning. Pruning methods can be classified based onwhen they are applied. Techniques identified as upstreampruning involve the pruning of the model before the fine-tuning stage. In contrast, downstream pruning methods arecharacterized by the simultaneous execution of pruningalongside the fine-tuning process. Accordingly, upstreampruning retains the adaptability of the pruned model formultiple tasks, ensuring its versatility. Conversely, down-stream pruning directs the pruned model to concentrate ona specific, well-defined task.

# 4.2 Pruning Methods for Medium-Size Language Mod-els

Language models, such as GPT-2 and BERT, are initiallytrained on extensive corpora and exhibit applicability acrossvarious downstream tasks after fine-tuning. Specifically, thepruning of language models distinguishes itself from thepruning methodologies employed in Convolutional NeuralNetworks (CNNs) or Recurrent Neural Networks (RNNs) inthree key aspects. First and foremost, the sheer magnitudeof parameters in language models surpasses that of CNNsor RNNs. For instance, the BERT-large model encompasses335 million parameters, whereas the parameters of a typicalRNN are in the range of tens of millions [99]. The increasednumber of parameters amplifies the temporal and compu-tational demands of the fine-tuning phase. Consequently,language model pruning necessitates addressing the chal-lenges posed by this substantial parameter abundance.Secondly, language models have the potential to undergofine-tuning for a multitude of downstream tasks. Certainupstream pruning methodologies necessitate the retentionof the language model’s capacity to function as a multi-task solver. Thirdly, transformer-based language modelsexhibit a distinctly different structural composition. Hence,in light of the model’s architecture, certain structured prun-ing methods may require reconfiguration to align with thestructure of the model. In conclusion, there exist specializeddesigns of pruning methodologies for language models thatare tailored to their unique characteristics, deviating fromconventional pruning approaches.

We will introduce these pruning techniques for medium-size language models in the following, including ap-proaches that are specially designed for Transformer-basedmodels and generic to plenty of models with differentarchitectures. In consideration of the fundamental featuresof pruning methods (i.e., the determination of what toprune and how to prune), we shall introduce these pruningmethods in the order of pruning unit and metric. Initially,we classify pruning methods into two primary components:unstructured and structured ones. Subsequently, based onthe sequence of pruning criteria, we will expound upon eachof the three pruning methods: magnitude-based pruning,loss-based pruning, and regularization.

# 4.2.1 Unstructured Pruning for Medium-Size LanguageModels

Unstructured pruning methods zero out non-essentialweights without any specific constraints. We will introduceunstructured pruning methods for medium-size languagemodels in a systematic order based on specific metrics,including magnitude-based pruning, loss-based pruning,and regularization.

# 1) Magnitude-based Pruning

Magnitude-based pruning, characterized by its simplic-ity and efficacy, incorporates the magnitudes of weights andactivation values into its pruning metrics. In this sectionon magnitude-based pruning for medium-size languagemodels, we find that all of the related methods exclusivelyfocus on the magnitudes of weights. Consequently, we willintroduce these magnitude-based pruning methods withweights.

Magnitude Pruning [98], recognized as the most com-monly utilized pruning method, has been examined in thecontext of medium-size language models [100], [101], [102],[103]. Gordon et al. [100] conducted a study focusing onthe compression of BERT through Magnitude Pruning. Thefindings reveal that approximately $3 0 - 4 0 \%$ of the weightsare non-essential and can be discarded without affectingBERT’s performance. Furthermore, fine-tuning BERT for aspecific task does not contribute to an enhancement in theultimately achievable sparsity ratio. This implies that BERTcan undergo pruning once during the pre-training phase,obviating the need for separate pruning for each task, allwhile maintaining performance integrity. Based on this,Prune Once for All [104] is to prune models once for alltasks before fine-tuning.

Magnitude pruning, characterized by the direct pruningof the model sparsity ratio to the target ratio, may result in asubstantial deterioration of model performance. Comparedto Magnitude Pruning, Gradual Magnitude Pruning (GMP)[105] introduces a sparsity ratio schedule, gradually reduc-ing the sparsity ratio throughout the pruning process. PruneOnce for All [104] and ${ \mathrm { G M P } } { \star }$ [106] are both implementationsof GMP specifically applied to language model pruning.Besides, $\mathrm { G M P } \star$ introduces an initial substantial pruning stepto better adapt to a high target sparsity (e.g., $9 7 \%$ ). This ap-proach allows for more recovery time in subsequent pruningsteps, ultimately leading to improved performance, outper-forming most pruning methods including Prune Once forAll [104].

# 2) Loss-based Pruning

While magnitude-based pruning is easy to implement,the magnitude alone may not accurately reflect the im-portance of weights in some instances. The magnitudeof certain weights may be diminutive, yet their contribu-tion remains essential [107]. Therefore, a more scientificallygrounded approach involves assessing these weights withinthe context of a specific task. The methods in this sectionadopt the loss-based pruning strategy tailored for medium-size language models. These approaches align with a morenuanced evaluation based on the performance. Given thatthe model’s training process is inherently geared towardsminimizing this loss, the loss undoubtedly stands out as themost reliable measure of the model’s performance.

The first major category of loss-based pruning methodsintegrates information about the gradients within the spe-cific metrics. The universal expression by which these meth-ods evaluate the importance of weights can be articulatedthrough the negative gradient-weight product, expressed asfollows:

$$
\mathbf {I} = - \mathbf {w} \nabla \mathcal {L} (\mathbf {w}) \tag {17}
$$

The first interpretation of this expression pertains to theweight change. The negative gradient direction of theweights signifies the direction in which the weights areintended to increase. Consequently, if the weight directionaligns with the direction of weight growth, it indicatesthe importance of that weight in the specific task, as thetask necessitates the continued increase in its magnitude.Alternatively, the second interpretation of this expressioncan be simplistically conceived as the first-order term of theTaylor expansion of loss alteration, with higher-order termsbeing disregarded.

Many methods have implemented their improvementsbased on this universal expression. Movement Pruning [108]accumulates multiple updates of the negative gradient-weight product. Accumulating such information aids inminimizing fluctuations during pruning. Among the first-order methods, Movement Pruning stands as a pioneeringone, upon which many extensions have been developed[109], [110]. To mitigate the substantial variability and un-certainty introduced by mini-batch sampling and intricatetraining dynamics, PLATON [111] employs a weight prun-ing strategy that considers both the importance and uncer-tainty associated with individual weights. The uncertaintyoriginates from changes in importance. To enhance stability,both importance and uncertainty undergo exponential mov-ing averaging. The final importance score of each weightis determined by the product of smoothed importance anduncertainty. Parameter-efficient Sparse Training (PST) [112]and LoRAPrune [113] add the magnitude of weight and theaccumulated negative gradient-weight product to derive thefinal importance score.

The second major category of loss-based pruning meth-ods integrates information about the second-order deriva-tive within the specific metrics. The variation of the loss,when employing the Taylor expansion and expanding up tothe second-order term while neglecting higher orders, canbe expressed in the following manner:

$$
\begin{array}{l} \mathcal {L} (\mathbf {w}) - \mathcal {L} \left(\mathbf {w} ^ {*}\right) \simeq \left(\mathbf {w} - \mathbf {w} ^ {*}\right) ^ {\top} \nabla \mathcal {L} \left(\mathbf {w} ^ {*}\right) \\ + \frac {1}{2} \left(\mathbf {w} - \mathbf {w} ^ {*}\right) ^ {\top} \mathbf {H} _ {\mathcal {L}} \left(\mathbf {w} ^ {*}\right) \left(\mathbf {w} - \mathbf {w} ^ {*}\right), \tag {18} \\ \end{array}
$$

where $\mathbf { H } _ { \mathcal { L } } ( \mathbf { w } ^ { * } )$ is the Hessian matrix. These methods arepost-training pruning methods and always prune a well-trained network $\mathbf { w } ^ { * }$ . Therefore, the gradient $\nabla \mathcal L ( \mathbf w ^ { * } )$ canbe neglected, getting a universal expression to represent theimportance of weights:

$$
\mathbf {I} = \frac {1}{2} \left(\mathbf {w} - \mathbf {w} ^ {*}\right) ^ {\top} \mathbf {H} _ {\mathcal {L}} \left(\mathbf {w} ^ {*}\right) \left(\mathbf {w} - \mathbf {w} ^ {*}\right). \tag {19}
$$

The Optimal Brain Damage (OBD) [114] and the Opti-mal Brain Surgeon (OBS) [115] represent the second-orderpruning approaches in early works. Both of them utilize theHessian of the loss function to selectively eliminate specific

parameters while minimizing the impact on loss. To stream-line calculations, both methods simplify the computation ofthe Hessian matrix to some extent. However, OBD computessolely the diagonal entries of the Hessian matrix, whereasOBS also considers the impact of off-diagonal entries. Thesemethodologies have served as inspiration for numeroussubsequent approaches. The Optimal BERT Surgeon [116]extends the principles of the OBS to the context of BERT,yielding better results when compared to some magnitude-based pruning methods [98], [104], [106] and the first-orderpruning methods [108], [111].

# 3) Regularization

In addition to the aforementioned methods, regular-ization techniques find many applications in medium-sizelanguage models. $L _ { 1 }$ and $L _ { 2 }$ regularization are popularmethods employed to counteract network overfitting. Bothintroduce a regularization term into the loss function. Be-sides, $L _ { 1 }$ regularization has the additional effect of inducingsparsity in weights. However, when it comes to directlypruning the network, $L _ { 1 }$ regularization is not always themost suitable choice. This is attributed to the fact that$L _ { 1 }$ regularization imposes more substantial penalties onlarger weights, deviating from the original pruning objec-tive of eliminating unimportant connections, where smallerweights are often situated.

Instead, $L _ { 0 }$ regularization [117] is a more versatile prun-ing method than $L _ { 1 }$ and $L _ { 2 }$ regularization. $L _ { 0 }$ regularizationincorporates the $L _ { 0 }$ norm of weights into the loss function.Similar to $L _ { 1 }$ and $L _ { 2 }$ regularization, $L _ { 0 }$ regularization pe-nalizes non-zero weights. However, it distinguishes itself byapplying equal penalties to all non-zero weights, aligningprecisely with the pruning objective of equitably penalizingall the existing connections.

The training objective of the pruning process for all threeof these regularizations can be expressed by the followingformula:

$$
\min  _ {\mathbf {w}, \mathbf {z}} \frac {1}{N} \sum_ {i = 1} ^ {N} \ell (\mathbf {w} \odot \mathbf {z}; (x _ {i}, y _ {i})) + \lambda \| \mathbf {w} \| _ {p}, \tag {20}
$$

where $\lambda$ represents the regularization factor, $\| \mathbf { w } \| _ { p }$ denotesthe $L _ { p }$ norm of the weights, and z is a binary mask indicat-ing whether the weights are pruned. Consequently, the $L _ { 0 }$norm of the weights can be equivalently represented by thesummation of binary masks, i.e., $\| \mathbf { w } \| _ { 0 } = \dot { \sum _ { i = 1 } ^ { M } } z _ { i }$ .

However, the discrete nature of z poses challenges forefficient gradient-based optimization. To this end, the hardconcrete distribution serves as an approximation to thebinary masks, allocating approximately half of its massto $\{ 0 , \ 1 \}$ and the remaining half to the interval (0, 1),thereby bridging the gap between discrete values 0 and 1with continuous probability mass, as shown in Fig. 4. Theformulation of the hard concrete distribution is as follows:

$$
\begin{array}{l} u \sim \mathcal {U} (0, 1), \\ s = \operatorname {S i g m o i d} \left(\left(\log u - \log (1 - u) + \log \alpha\right) / \beta\right) \tag {21} \\ \end{array}
$$

$$
\begin{array}{l} \bar {s} = s (\zeta - \gamma) + \gamma , \\ z = \min  (1, \max  (0, \bar {s})), \\ \end{array}
$$

where $\mathcal { U } ( \cdot )$ is a uniform distribution, $\log \alpha$ is the locationparameter, $\beta$ is the temperature parameter, $( \gamma , \zeta )$ is the“stretch” interval with $\gamma < 0$ and $\zeta > 1$ . Given that the

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-11/0a775777-6cdd-4dbe-a304-295f64b9a1ad/c2055ccc84db8b80589292266b15b2d6138564fe423753b7be5cfde05e8ce173.jpg)



Fig. 4: The approximate probability density histogram ofhard concrete distribution by using Monte Carlo simulation.The parameters of this hard concrete distribution are $\log \alpha =$0, $\beta \stackrel { - } { = } 0 . 5$ , $\gamma = - 0 . 1 ,$ and $\zeta = 1 . 1$ . Under this specificationthe hard concrete distribution assigns, roughly, half of itsmass to $\{ 0 , 1 \}$ and the rest to (0, 1).


reparameterized variable $z$ is not strictly binary after train-ing, many pruning methods adopt a threshold to discretize$z$ into binary values in the end. For values of $z$ below thethreshold, the value is set to 0, while for values above thethreshold, the value is set to 1.

While $L _ { 0 }$ regularization finds broader applications inpruning, $L _ { 1 }$ regularization also has some pertinent usecases, and certain methods strive to enhance $L _ { 1 }$ regular-ization. For example, Reweighted Proximal Pruning (RPP)[118] builds upon $L _ { 1 }$ regularization and introduces im-provements and refinements. RPP comprises reweighted $L _ { 1 }$regularization and the proximal operator. The reweighted$L _ { 1 }$ regularization dynamically reallocates penalty factors,assigning greater penalties to weights approaching zero.The proximal operator facilitates the separation of the spar-sity pattern search and the back-propagation-based gradientupdate of the training loss, enabling an easier sparse patternsearch.

# 4) Others

Among the unstructured pruning methods discussedabove, numerous approaches demonstrate an ability touphold satisfactory model performance even with highsparsity. However, they encounter challenges in achievingefficient inference speedup due to the irregular nature of thesparse matrices they generate. To address this predicament,unstructured pruning methods can be integrated with N:Msparsity [119].

The principle underlying N:M sparsity mandates thatwithin each group of $M$ consecutive weights in the neuralnetwork, no more than $N$ weights should exhibit non-zerovalues. This implies that within each group of $M$ consec-utive weights, there are $N - M$ weights with zero values.Thus the underlying hardware can compress the regularlyoccurring zero values within it. This kind of compressionrelies on unique architectures of the hardware, such assparse tensor cores. For instance, the Nvidia Ampere A100is equipped with sparse tensor cores to accelerate the 2:4sparsity.

For N:M sparsity, the pruning metric is not a restricted

factor. It can be seamlessly integrated with unstructuredpruning methods, providing the inference speedup thatpure unstructured methods may lack. For example, thedetermination of the sparsity pattern can be initially pred-icated on the magnitudes of the weights [120]. Serving asa generic sparsity methodology, 2:4 sparsity demonstrates anotable twofold acceleration in computational speed with-out compromising performance.

# 5) Discussion

Among all these unstructured pruning methods formedium-size models, the Optimal BERT Surgeon [116]demonstrates superior performance compared to variousmagnitude-based pruning methods [98], [104], [106] and thefirst-order pruning methods [108], [111] in the conductedexperiments [106], [116].

Nonetheless, Magnitude Pruning [98] remains the mostwidely adopted pruning method. Because it is simple toimplement, yet achieves competitive results with manyintricate methods [121]. Crucially, the pruning process ofMagnitude Pruning operates independently of any specificdataset, thereby addressing challenges in some scenarioswhere datasets may be unavailable.

# 4.2.2 Structured Pruning for Medium-Size Language Mod-els

Indeed, numerous unstructured pruning methods havedemonstrated the capability to achieve a high sparsity ratiowhile maintaining performance levels comparable to theirdense counterparts. However, it’s noteworthy that unstruc-tured sparse patterns do not necessarily lead to inferencespeedup on normal hardware. Consequently, there is an in-creasing focus on research dedicated to structured pruning.

In the context of structured pruning methodologies ap-plied to medium-size language models, the selection ofappropriate pruning units assumes significance alongsidethe choice of pruning metric. Pruning units commonlyconsidered encompass attention heads, FFN neurons, hid-den dimensions, and etc. Notably, employing architecture-related structures as pruning units tends to yield morefavorable outcomes compared to structures unrelated tomodel architecture, such as weight blocks.

This observed superiority may be attributed to thepreservation of fundamental principles inherent in themodel’s construction when reducing architecture-relatedstructures. For instance, after the pruning of attention heads,the resultant model retains the essential characteristics of atransformer-based model, featuring a reduced number ofattention heads.

In the following, we will delve into the realm of struc-tured pruning, encompassing magnitude-based pruning,loss-based pruning, and regularization techniques.

# 1) Magnitude-based Pruning

Intuitively, the aggregation of weight magnitudes forpruning units serves as a meaningful representation ofimportance, which is widely applicable to convolutionalkernels in CNNs. Similarly, it can be extended to medium-size language models. For instance, the weight magnitudescould be aggregated with $L _ { 2 }$ norm to represent the corre-sponding importance in attention heads, FFN neurons [122],and weight blocks [123]. The less important structures arethen removed based on the order of their importance scores.

# 2) Loss-based Pruning

Within loss-based pruning methodologies, considerableattention has been directed towards the exploration andanalysis of attention heads [124], [125], [126], [127]. Thisfocus emanates from their proclivity to become redundant,and that the rest of the heads frequently could demonstratean aptitude for assuming the functional roles previouslycarried out by the pruned heads. Michel et al. [124] proposedan iterative pruning approach based on head importancescores. The attention heads are covered with binary maskvariables. Thus, the head importance scores are computedthrough the examination of gradients on the binary maskvariables. The results indicated that $2 0 – 4 0 \%$ of heads Trans-former heads could be pruned without significantly com-promising test accuracy on the target task.

However, Differentiable Subset Pruning (DSP) [125]demonstrated that Michel et al. [124] significantly under-estimated the number of Transformer heads that could bepruned. The experiments showed that DSP could prune upto $9 0 \%$ of heads without causing much degradation in testperformance. (Pruning up to $9 0 \%$ of heads means around$2 0 \%$ of the model size shrinkage as the parameters of headsare just part of the whole model.) DSP treats Transformerhead pruning as a subset selection problem. To ensure thedifferentiability of the subset pruner, the Gumbel–Softmaxtrick [128] and its extension to subset selection is applied toDSP. The results indicated superior accuracy and inferencespeedup of DSP compared to other head pruning methods[124], [129].

In addition to attention head pruning, Block Move-ment Pruning [130] is a block pruning method. It extendsstructured methods by considering blocks of any size andintegrates these structures into the Movement Pruning [108].The matrix within the model undergoes partitioning intofixed-sized blocks, with larger block sizes yielding greaterinference speedup. Furthermore, the combination of thisapproach with the pruning of neurons in the FFNs results inthe best overall performance. Similarly, numerous method-ologies prune neurons in the FFNs and attention headssimultaneously [131], [132], [133], [134], [135], [136].

In addition to the above methods designed for theTransformer structures, some structured pruning methodscan be generalized because the pruning units in them areneurons [137], [138], [139]. For instance, Low-Rank andSparse approximation (LoSparse) [137] prunes the weightmatrix in neuron level (i.e., the columns of weight matrix).Considering the sensitivity of parameters defined in PLA-TON [111], the importance of each neuron is defined by thecumulative sensitivity of parameters within a given column.

# 3) Regularization

In addition to the loss-based pruning methods, regu-larization methods constitute another category within thespectrum of structured pruning techniques applicable tomedium-size language models. Diverging from unstruc-tured pruning approaches, the regularization term in struc-tured pruning encompasses binary masks associated withspecific structural components, as opposed to individualweights. Except for the pruning units, other details closelyresemble those in unstructured pruning.

Nevertheless, among these regularization methods, $L _ { 0 }$regularization stands out as the most extensively employed

technique. The main variability among these $L _ { 0 }$ regular-ization methods resides in their respective approaches tothe selection of pruning units. Voita et al. [129] introduced$L _ { 0 }$ regularization to attention head pruning, specificallyselecting a subset of attention heads. McCarley et al. [140]incorporated $L _ { 0 }$ regularization to prune attention heads andFFN neurons. Factorized Low-rank Pruning (FLOP) [141]integrates Low-Rank Factorization with $L _ { 0 }$ regularization.This methodology involves the reparameterization and fac-torization of the matrix W into the product of two smallermatrices, denoted as $\mathbf { W } = \mathbf { P } \mathbf { Q } ,$ , where $\boldsymbol { \mathsf { p } } _ { k }$ and $\mathbf { q } _ { k }$ representthe $k$ -th column of P and $k \mathrm { . }$ -th row of $\mathbf { Q }$ respectively. Thepruning unit is the combination of $\boldsymbol { \mathsf { p } } _ { k }$ and $\mathbf { q } _ { k }$ . Additionally,an augmented Lagrangian method is introduced to regulatethe sparsity ratio in the context of FLOP. Coarse- and Fine-grained Pruning (CoFi) [142] jointly prunes coarse-grainedand fine-grained modules using $L _ { 0 }$ regularization, includ-ing attention and FFN layers, individual attention heads,FFN neurons, and hidden dimensions for Transformer-based models. Notably, the mask over the hidden dimensionis shared across all Transformer layers and an augmented la-grangian method is adapted. By combining with a layerwisedistillation approach, CoFi achieves models with more than$1 0 \times$ speedups while exhibiting only a marginal decrease inaccuracy in the conducted experiments.

In addition to $L _ { 0 }$ regularization, $L _ { 1 }$ regularization alsogets relevant research. SIMPLE [143] introduces $L _ { 1 }$ regu-larization to structured pruning, encompassing attentionheads, intermediate neurons of the FFN, and the hiddendimension as compressible components. The mask overthe hidden dimension is shared across layers, akin to theapproach employed in CoFi [142]. Through the learning ofmasks for these compressible components via a sparsity-induced objective, various-sized pruned models can beobtained. These pruned models can subsequently be fine-tuned with a causal distillation objective to enhance perfor-mance.

# 4) Others

Beyond the classification based on metrics, certain struc-tured pruning methods exhibit notable similarities whentheir designated pruning units are identical.

The first class among other structured pruning is layerpruning [144], [145], [146]. The aforementioned pruningunits, such as attention heads and neurons, are characterizedby their relatively diminutive scale, necessitating a moredetailed pruning scheme to determine which should bepruned. Conversely, when dealing with substantially largerpruning units, such as entire layers, numerous methodolo-gies tend to engage in direct experimentation with multi-ple pruning schemes before determining the most effectivenetwork configuration. This practice stems from the lowertesting costs associated with a smaller number of layers.

In addition to layer pruning, there is a body of re-search dedicated to token pruning [147], [148], [149], whichdoes not alter the underlying network architecture. Tokenpruning involves the removal of unimportant tokens froma sequence during inference to reduce computational re-quirements. Learned Token Pruning (LTP) [148] representsa straightforward and effective approach to adaptively re-move unimportant tokens as an input sequence traversesthrough transformer layers. The pruning metric for each

token is determined by the sum of normalized attentionprobability from the Transformer block.

Extending beyond the pruning units previously men-tioned, structured pruning encompasses a myriad of di-verse units. For instance, Spectral-Normalized Identity Prior(SNIP) [150] employs a strategy to prune attention andFFN sublayers by transforming residual connections intostrict identity mappings. SNIP sets specific thresholds foractivation vectors, and those falling below the thresholdsresult in the pruning of residual blocks (i.e., the attentionand FFN sublayers).

# 4.3 Pruning Methods for LLMs

In the last section, we introduced the pruning methods formedium-size language models with parameters numberingless than 1 billion. Most of these methods adopt full fine-tuning after pruning to improve the performance. However,as the parameters increase, full fine-tuning becomes moredifficult or even infeasible. This discrepancy underscoresa significant challenge in the field of research dedicatedto pruning techniques tailored specifically for LLMs. Tohandle this problem, on the one hand, certain pruningmethodologies opt to incorporate parameter-efficient tuningtechniques to reduce fine-tuning costs. On the other hand,alternative approaches abandon the fine-tuning process,relying on an optimized pruning procedure will inherentlylead to retained model performance. The viability of thesealternative approaches is partly attributed to the huge num-ber of parameters in LLMs. The higher number implies ahigher likelihood of redundancy within the model.

In this section, we will introduce the pruning methodsfor LLMs, mirroring the sequence established in the sectiondevoted to pruning methods for medium-size languagemodels. Pruning methods for LLMs adhere to a parallelapproach to those employed for medium-size languagemodels, with some distinctions in certain methods primarilyarising in omitting the fine-tuning process. To facilitatea more comprehensive comparison of these methods, weconsolidate the characteristics of these pruning methods, asshown in TABLE 3.

# 4.3.1 Unstructured Pruning for LLMs

Attributed to the greater capacity of unstructured pruningmethods to preserve model performance compared to struc-tured alternatives, all of the unstructured pruning method-ologies in this section for LLMs adopt an approach ofeschewing the fine-tuning process as shown in TABLE 3. Theexperiments have demonstrated that these methodologiescan attain a sparsity ratio of $5 0 \%$ with a relatively modestcompromise in model performance.

The two pioneer unstructured pruning methods forLLMs are SparseGPT [154] and Wanda [151], which becomethe baselines for many subsequent methods for compari-son. The subsequent unstructured pruning methods demon-strate their capability to outperform SparseGPT and Wandaacross various NLP tasks, thereby attaining superior results.Though unstructured pruning methods get hardly inferencespeedup, they can easily be combined with N:M sparsity[119] to accelerate inference speed, which is also experi-mented in SparseGPT and Wanda.

These unstructured pruning methods require minimalcalibration data. The minimal calibration data is for a singleforward pass of the model, specifically aiming at acquiringactivation values or gradients to calculate the importance ofweights, which remains a contributing factor to the outcomeof the pruning [166].

In the following, we will introduce these unstructuredpruning methods in LLMs in the order of pruning met-rics. In this investigation, no regularization-related methodshave been identified, thus this section will be divided intointroductions of methods based on magnitude and methodsbased on loss.

# 1) Magnitude-based Pruning

When directly applying Magnitude Pruning [98] toLLMs, the outcomes are not very competitive even withparameter-efficient fine-tuning strategies [167], [168]. There-fore, in magnitude-based pruning methods, compared toonly using the magnitude of weights as the pruning met-ric in medium-size language models, more magnitude-based pruning methods in LLMs combine the magnitudeof weights and activate values as the pruning metric. Forinstance, Wanda [151] and RIA [152] use the magnitude ofweight and activation metric. In addition to the magnitudeof weight and activation, E-Sparse [153] also introduces theinformation entropy into the metric.

Wanda (Pruning by Weights and activations) [151] intro-duces a novel pruning metric, considering both the magni-tude of weights and activate values. The motivation is thatthe significance of weights should not solely be evaluated inisolation but rather in consideration of its product with thecorresponding activation value. To illustrate, let’s considera fully connected layer with weights represented by Wwith dimensions $( C _ { o u t } , C _ { i n } )$ . In the context of languagemodels, this linear layer receives input activation X withdimensions $( N \times L , \dot { C } _ { i n } )$ , where $N$ and $L$ denote the batchand sequence dimensions respectively. For each weight, itsimportance is quantified as the product of its magnitude andthe corresponding input feature norm. Concretely, the score$\mathbf { S } _ { i j }$ for the weight $\mathbf { W } _ { i j }$ is defined as:

$$
\mathbf {S} _ {i j} = \left| \mathbf {W} _ {i j} \right| \cdot \left\| \mathbf {X} _ {j} \right\| _ {2}, \tag {22}
$$

where $\| \mathbf { X } _ { j } \| _ { 2 }$ evaluates the $L _ { 2 }$ norm of $j$ -th features aggre-gated across $N \times L$ different tokens. Remarkably, the resultsindicate that Wanda achieves comparable performance toSparseGPT but in a significantly shorter time.

Similar to Wanda [151], RIA (Relative Importance andActivations) [152] also jointly considers the weight andactivation. The primary distinction lies in its approach toalleviating channel corruption (i.e., the rows and columnsof the weight matrix pruned integrally). RIA replaces themagnitude of weights with relative importance. This rela-tive importance is calculated as the magnitude of individualweights divided by the sum of the magnitude of weightsin their corresponding row and column. Therefore, thecomparison among different rows and columns becomesrelatively equitable by utilizing the relative importance, mit-igating potential biases introduced by the variations in theirmagnitudes. RIA can be further combined with channelpermutation, which maximally preserves important weightsunder N:M sparsity to get practical speed-up on specifichardware.


TABLE 3: A summary of various pruning methods for LLMs.


<table><tr><td>Methods</td><td>Unit</td><td>Metric</td><td>Iterative/One-shot</td><td>Finetuning</td><td>Global/Local</td></tr><tr><td>Wanda [151]</td><td>Unstructured</td><td>Magnitude-based</td><td>One-shot</td><td>No</td><td>Local</td></tr><tr><td>RIA [152]</td><td>Unstructured</td><td>Magnitude-based</td><td>One-shot</td><td>No</td><td>Local</td></tr><tr><td>E-Sparse [153]</td><td>Unstructured</td><td>Magnitude-based</td><td>One-shot</td><td>No</td><td>Local</td></tr><tr><td>SparseGPT [154]</td><td>Unstructured</td><td>Loss-based</td><td>One-shot</td><td>No</td><td>Local</td></tr><tr><td>ISC [155]</td><td>Unstructured</td><td>Loss-based</td><td>One-shot</td><td>No</td><td>Local</td></tr><tr><td>GBLM-Pruner [156]</td><td>Unstructured</td><td>Loss-based</td><td>One-shot</td><td>No</td><td>Local</td></tr><tr><td>PGZ [157]</td><td>Unstructured</td><td>Loss-based</td><td>One-shot</td><td>No</td><td>Local</td></tr><tr><td>FLAP [158]</td><td>Structured</td><td>Magnitude-based</td><td>One-shot</td><td>No</td><td>Global</td></tr><tr><td>SliceGPT [159]</td><td>Structured</td><td>Magnitude-based</td><td>One-shot</td><td>PEFT</td><td>Local</td></tr><tr><td>LLM-Pruner [160]</td><td>Structured</td><td>Loss-based</td><td>One-shot</td><td>PEFT</td><td>Global</td></tr><tr><td>LoRAShear [161]</td><td>Structured</td><td>Loss-based</td><td>Iterative</td><td>PEFT</td><td>Global</td></tr><tr><td>APT [162]</td><td>Structured</td><td>Loss-based</td><td>Iterative</td><td>PEFT</td><td>Global</td></tr><tr><td>Sheared LLaMA [163]</td><td>Structured</td><td>Regularization</td><td>One-shot</td><td>Yes</td><td>Local</td></tr><tr><td>Compresso [164]</td><td>Structured</td><td>Regularization</td><td>Neither</td><td>PEFT</td><td>Global</td></tr><tr><td>LLM Surgeon [165]</td><td>Both</td><td>Loss-based</td><td>Iterative</td><td>PEFT</td><td>Global</td></tr></table>

In addition to the magnitude of weight and activationas Wanda and RIA, E-Sparse (Entropy-based Sparsity) [153]introduces information entropy from hidden state featuresinto the pruning metric. The entropy serves as a measure ofinformation richness, with higher values indicating richerinformation. Consequently, entropy is incorporated along-side standard weight magnitude and input feature normin the pruning metric, enhancing the evaluation of channelinformation activation.

# 2) Loss-based Pruning

In loss-based approaches, it is observed that the prun-ing metrics involve the first or second-order derivatives ofweights with respect to the loss. The second-order methodsdiscussed in this subsection are all inspired by two earliersecond-order loss-based pruning methods, namely, OptimalBrain Damage (OBD) [114] and Optimal Brain Surgeon(OBS) [115].

SparseGPT [154], a second-order pruning method, incor-porates OBS [115] technique into the GPT-family models.It is the first pruning method that works efficiently atmodels with $1 0 \mathrm { - } 1 0 0 \mathrm { + }$ billion parameters. The SparseGPTpruning methodology is delineated by two main compo-nents: mask selection and weight reconstruction processes.Initially, the mask selection identifies weights for pruningbased on a metric, such as weight magnitude. Subsequently,the unpruned weights undergo optimization using the OBSmethod to reconstruct the compressed model (i.e., updatethe remaining parameters) to compensate for the prunedweights. The pruning procedure in SparseGPT requires min-imal calibration data. These data undergo a single forwardpropagation, during which the unpruned weights are up-dated only once. The results of this approach demonstratethat LLMs can be compressed to high sparsity throughweight pruning in a single pass, without necessitatingthe fine-tuning process. Importantly, this compression isachieved with a low loss of accuracy, as assessed by perplex-ity and zero-shot performance metrics. Similarly, the LLMSurgeon [165] extents OBS but is generic for unstructuredand structured pruning.

Building upon the concepts of OBS and OBD, Shao etal. [155] introduced a novel pruning metric termed the

Improved Saliency Criterion (ISC). ISC is devised by addingthe metrics derived from OBS and OBD directly. This newmetric aims to provide a comprehensive and refined as-sessment of the importance of model parameters for thepruning process. In addition to proposing ISC, Shao et al.put forward to allocate sparsity ratio individually to eachmatrix. In this way, pruning targets are selected adaptivelywithin each weight matrix.

In addition to the aforementioned second-order meth-ods, there has been corresponding research into first-order methods [156], [157]. Gradient-based Language ModelPruner (GBLM-Pruner) [156] is a first-order pruningmethod. The importance of weights is defined by the prod-uct with the magnitude of weights and the normalizationof the corresponding gradients across different samples,which can be seen as an extension of the traditional first-order method (i.e., gradient-weight product). Furthermore,the feature activations can be integrated into the pruningmetric to enhance performance.

# 4.3.2 Structured Pruning for LLMs

In contrast to unstructured pruning, structured pruning isnot constrained by hardware limitations, enabling the real-ization of inference acceleration on conventional hardwarefollowing the pruning process. However, these methodsmight result in more performance degradation than un-structured ones due to the alteration of network structures,necessitating a fine-tuning process to recover performance.Therefore, while fine-tuning is abandoned in unstructuredpruning for LLMs, it is widely employed in structuredpruning for LLMs but in a parameter-efficient way. Similarto unstructured pruning, structured pruning for LLMs hasits pioneer method, LLM-Pruner [160], which serves as abaseline for subsequent methods and facilitates meaningfulcomparisons.

The discussion of these structured pruning methods forLLMs will be presented in the following section. Similarly,we will introduce these structured pruning methods inLLMs in the order of pruning metrics, including magnitude-based pruning, loss-based pruning, and regularization.

# 1) Magnitude-based Pruning

Magnitude-based pruning methods for LLMs considerrows or columns as pruning units [158], [159], [169]. Forinstance, the pruning units of FLuctuation-based AdaptiveStructured Pruning (FLAP) [158] are columns. The impor-tance score of each column of the weight matrix is mea-sured by the ”fluctuation metric”. This metric is the samplevariance of each input feature which is weighted with thesquared norm of the corresponding column of the weightmatrix. Furthermore, in its pursuit to obviate the necessityfor fine-tuning, FLAP incorporates bias compensation mech-anisms aimed at mitigating the adverse effects stemmingfrom the removal of components.

# 2) Loss-based Pruning

In the realm of loss-based structured pruning methodsapplied to LLMs, gradients remain pivotal information, akinto their significance in medium-size models. The followingmethods utilize gradient information in different ways [160],[161], [162], [170], such as defining pruning structures, se-lecting pruning targets, and etc. The most notable departureof these methods from traditional approaches lies in theiravoidance of predefined pruning units (e.g., attention heads,neurons). Instead, some of these methods dynamically iden-tify and designate pruning units.

For instance, LLM-Pruner [160] removes non-criticalcoupled structures during the pruning process. These cou-pled structures are automatically identified and extractedthrough the definition of structure dependency (i.e., connec-tion dependencies between neurons). A coupled structurecomprises a group of weights. The importance of individualweights is formulated as the change in loss, expanded usingTaylor expansion to the second order. The diagonal of theHessian matrix in the second-order term is approximated bythe Fisher information matrix using first-order information.Ultimately, the importance of a group of weights is aggre-gated through summation, production, or other methods todetermine the group’s overall importance. After evaluatingthe importance of each group, those with lower importanceare pruned based on a predefined pruning ratio. The fine-tuning process in LLM-Pruner applies some parameter-efficient tuning techniques, such as LoRA. This facilitatesrapid and effective fine-tuning of pruned models using asmall amount of data. The experimental results showcasethat when $2 0 \%$ of the parameters are removed, the prunedmodel maintains the performance of the majority of the orig-inal model. However, a more aggressive pruning strategy,involving the removal of $5 0 \%$ of the parameters, results in asubstantial decline in model performance. This observationalso underscores the difficulty of achieving high sparsityratios through structured pruning while maintaining modelperformance.

Similar to LLM-Pruner, LoRAShear [161] discovers theminimal removal structures in the dependency graph. How-ever, LoRAShear specifically constructs dependency graphsover LoRA modules, considering their learnable nature.The analysis of knowledge distribution is then utilized toidentify crucial structures, marking them as unprunable.A distinctive feature of LoRAShear is the introduction ofLoRA Half-Space Projected Gradient (LHSPG) for progres-sive structured pruning. LHSPG leverages information fromLoRA modules to identify and remove redundant structureswhile preserving the knowledge stored in the important

structures. This is achieved through the projection of redun-dant structures onto zero, transferring the knowledge to thecrucial structures.

In contrast to the manual design of pruning features,Ji et al. [170] proposed a novel approach by employinga non-neural model, specifically a gradient boosting de-cision tree (GBDT), as an accuracy predictor. The use ofthis accuracy predictor enables further optimization of thesearch space and search process for identifying the optimalpruned model automatically. By training the GBDT as anaccuracy predictor, the model gains the ability to assess andpredict the impact of different pruning configurations on theaccuracy of the neural network, facilitating more efficientand automated selection of the optimal pruned model.

# 3) Regularization

In the context of regularization methods applied toLLMs, contemporary approaches predominantly adhere tothe principles established for earlier medium-size languagemodels, incorporating some generic refinements and opti-mizations.

Sheared LLaMA [163] can be viewed as an extensionof CoFi [142]. This approach involves the joint pruning ofcoarse-grained and fine-grained modules in Transformer-based models using $L _ { 0 }$ regularization. The modules sub-jected to pruning include layers, individual attention heads,FFN neurons, and hidden dimensions as in CoFi. ShearedLLaMA introduces two novel and significant components.The first component is targeted structured pruning, whichframes pruning as a constrained optimization problem. Thisformulation aims to learn pruning masks that search fora subnetwork matching a pre-specified target architecturewhile maximizing performance. The second component isdynamic batch loading, a strategy that loads training datafrom each domain in proportion to its rate of loss reduc-tion. This approach efficiently utilizes data and acceleratesoverall performance improvement during training. In a full-resource setup, Sheared LLaMA achieves compact counter-parts that outperform models of equal sizes trained fromscratch.

Compresso [164] integrates LoRA into the $L _ { 0 }$ regulariza-tion. The $L _ { 0 }$ regularization is employed to optimize binarymasks that cover modules including heads, FFN intermedi-ate neurons, and hidden dimensions. Simultaneously, modelparameters are updated through LoRA in the instructiontuning process. An innovative aspect of Compresso is theintroduction of a collaborative pruning paradigm where thepruning algorithm and target LLM work together through acollaborative prompt to learn the optimal pruning decisionsduring the instruction tuning process. The prompt explainsthe concept of pruning and its purpose, informs the LLMthat it is undergoing pruning, and encourages the LLMto better adapt to the pruning process. By incorporatingthis informative prompt, Compresso aims to enhance theLLM’s understanding and cooperation during pruning, con-tributing to improved performance and adaptation to themodified model structure.

# 4.4 Other Topics for LLM pruning

# 4.4.1 Enhancing Pruning Efficacy for LLMs

Several auxiliary techniques have been developed to en-hance the efficacy of pruning methods tailored for LLMs,

including the sparsity ratios tailored for subregions [171],[172], post-pruning fine-tuning methods [89], [167], [173],[174], [175], and hardware optimization [176], [177] Whilenot constituting a novel pruning method, these auxiliarytechniques can readily be integrated with existing pruningmethods for LLMs to enhance overall pruning outcomes.

One such method of tailored sparsity ratios is OutlierWeighed Layerwise sparsity (OWL) [171]. The experimentsin OWL indicate that the appropriate layerwise sparsityratios have a strong correlation with the emergence ofoutliers. Therefore, the sparsity ratio of OWL is directlyproportional to the outlier ratio observed within each layer.Consequently, in contrast to the prevailing LLM pruningstrategies that uniformly apply sparsity levels across alllayers, OWL introduces a customized set of non-uniformlayerwise sparsity ratios. Another approach of post-pruningfine-tuning methods is Dynamic Sparse No Training [174],which introduces a training-free fine-tuning method forsparse LLMs. This allows for slight updates to sparse LLMs,enabling further refinement without the need for a completefine-tuning process. Without the expensive backpropaga-tion, Dynamic Sparse No Training minimizes the recon-struction error between the dense and sparse LLMs, in thefashion of performing iterative weight pruning and growingon top of sparse LLMs.

The experimental results demonstrate that these tech-niques can significantly improve the performance of existingpruning methods, such as Wanda and SparseGPT. Thesefindings suggest that the potential enhancements to theperformance of pruning can be achieved through variousmeans unrelated to the cores of the pruning methods.

# 4.4.2 Future Works of Pruning for LLMs

While the field of pruning for LLMs has yielded fruitfulresults, it continues to grapple with significant challenges.Two primary issues stand out as particularly crucial.

Firstly, the integration of pruning with other methodolo-gies, such as quantization [154] and knowledge distillation[163], is essential for achieving competitive performance.Relative to the achievements of pruning in the domain ofvisual models in the past, the current outcomes in LLMpruning are comparatively less satisfactory. Therefore, a piv-otal challenge lies in augmenting the inherent effectivenessof the pruning method, ensuring its proficiency even whenemployed independently.

Secondly, the fine-tuning cost is a significant challengein the pruning of LLMs. Many pruning methods for LLMsadopt one-shot pruning without fine-tuning to minimizethe computational burden. Alternatively, some approachesincorporate parameter-efficient tuning techniques to reducetraining costs. However, such strategies inevitably compro-mise the performance of the pruned model. Researchersand practitioners in the field must persist in addressingthe challenge of the inability to execute full fine-tuning,particularly when dealing with LLMs aiming to enhance theperformance of pruning.

In conclusion, addressing these challenges is imperativefor advancing the effectiveness and practicality of pruningtechniques.

# 5 KNOWLEDGE DISTILLATION

Knowledge Distillation (KD) is a common technique forcompressing and speeding up models. The specific im-plementation process involves transferring the knowledgeacquired by a complex teacher model to a simpler student model,thereby enabling a more concise and efficient representationof the teacher model’s knowledge.

In Section 5.1, we will introduce some fundamental con-cepts of knowledge distillation and provide a brief classifica-tion of knowledge distillation methods. Then we will sum-marize various knowledge distillation methods employingmedium-size language models (the language models witharound 1 billion parameters) in Section 5.2, and we will clas-sify them into three groups based on whether distillation oc-curs during the pretraining phase, the finetuning phase, orboth. We finally provide a detailed overview of knowledgedistillation for large language models (the language modelswith over 1 billion parameters), categorizing them as black-box distillation and white-box distillation in Section 5.3.

# 5.1 Basic Concepts

Understanding the core of knowledge distillation involvesanswering three questions: what is knowledge, betweenwhom is knowledge transmitted, and how is knowledgetransmitted. Knowledge, in simple terms, is summarizedas the abilities the models possess (classification, reasoning,etc). In the distillation process, the source of knowledge isthe teacher model, and the recipient of knowledge is thestudent model. In other words, a well-trained teacher isessential, and our goal is to enable the student to acquireor reinforce the abilities the teacher possesses. However, thekey lies in how knowledge is transmitted. The pioneers ofknowledge distillation, Hilton et al. [178], first used the out-puts of the teacher and student’s softmax layers to transmitknowledge. They designed the following loss function totrain the student model, thereby achieving the transfer ofknowledge:

$$
L = \alpha \cdot L _ {D} \left(p \left(z _ {t}, T\right), p \left(z _ {s}, T\right)\right) + (1 - \alpha) \cdot L _ {S} \left(y, p \left(z _ {s}, T\right)\right) \tag {23}
$$

where $L _ { D } ( p ( z _ { t } , T ) , p ( z _ { s } , T ) )$ represents the difference inoutput of the softmax layers between the student andteacher, $L _ { S } ( y , p ( z _ { s } , T ) )$ represents the difference betweenthe output of the student’s softmax layers and the ground-truth labels. Both of them utilize the cross-entropy loss. $\alpha$represents the weight coefficient, and the specific expressionfor $p _ { i }$ is as follows:

$$
p _ {i} = \frac {\exp \left(z _ {i} / T\right)}{\Sigma_ {j} \exp \left(z _ {j} / T\right)} \tag {24}
$$

where T is employed to amplify the impact of incorrectlabels on the transmission of knowledge, thereby enablingthe student model to acquire more knowledge from a singlesample.

Subsequent researchers have employed a variety ofmethods to achieve knowledge transfer, primarily fallinginto the following four categories: logit-based KD, feature-based KD, relation-based KD and black-box KD. In Fig. 5,we also provide a brief overview of these distillation meth-ods and their relationships.

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-11/0a775777-6cdd-4dbe-a304-295f64b9a1ad/db7c94d873fbe994596ad1dc87d32b6792f810ad4300bdffda0aea97edd5f54c.jpg)



Fig. 5: Taxonomy of knowledge distillation


1) Logit-based KD. As the name suggests, logit-basedKD is a distillation paradigm that involves the transfer ofknowledge using the logits from the teacher model. We canwrite down the general form of the logit-based knowledgedistillation loss function:

$$
L _ {\text {l o g i t}} = \mathcal {L} \left(p \left(z _ {t}\right), p \left(z _ {s}\right)\right) \tag {25}
$$

where $\mathcal { L } ( \cdot )$ indicates the cross-entropy loss [178], Kullback-Leibler divergence (KLD) loss [179] and so on.

Clearly, Hilton et al.’s method is a example of logit-basedknowledge distillation.

2) Feature-based KD. Due to the limited knowledge ac-quired by the student in logit-based knowledge distilla-tion, researchers aim for better emulation of the teacher’sbehavior. Hence, they introduced feature-based knowledgedistillation. Specifically, this involves matching the outputsof intermediate layers in both the student and teacher mod-els, requiring the student not only to know the results butalso to understand the underlying processes. The followingis the general form of the loss function for feature-basedknowledge distillation:

$$
L _ {\text {f e a t u r e}} = \mathcal {L} \left(\left(f _ {t} (x), r \left(f _ {s} (x)\right)\right)\right) \tag {26}
$$

where $f _ { t } ( \cdot )$ and $f _ { s } ( \cdot )$ represent the feature maps of theteacher model and the student model. $\mathcal { L } ( \cdot )$ is the functionused to fit features, and $r ( \cdot )$ is applied to make feature mapsof the teacher model and the student model have the sameshape.

For example, FitNet [180] leverages feature maps fromintermediate layers of both the teacher and student modelsto adjust the parameters of the student model. It also usesmean squared error (MSE) along with a learnable matrix as$\mathcal { L } ( \cdot )$ and $r ( \cdot )$ .

3) Relation-based KD. Furthermore, researchers aim forthe student to learn how the teacher handles relationshipsbetween different data, leading to the proposal of relation-based knowledge distillation. This relationship is primarilymanifested in two aspects: the relationship between outputsat different layers for the same sample and the relationship

between outputs for different samples. The general form ofits loss function is as follows:

$$
L _ {\text {r e s p o n s e}} = \mathcal {L} \left(\left(f _ {t} \left(t _ {i}, t _ {j}\right), f _ {s} \left(s _ {i}, s _ {j}\right)\right) \right. \tag {27}
$$

where $t _ { i } , t _ { j }$ and $s _ { i } , s _ { j }$ are feature representations from theteacher model and the student model. They can representoutputs from different layers or outputs from different sam-ples. $f _ { t } ( \cdot )$ and $f _ { s } ( \cdot )$ represent the similarity functions.

For example, FSP [181] uses feature maps of the samesize as feature representations, and employs Gram matrixand MSE as $f ( \cdot )$ and $\mathcal { L } ( \cdot )$ .

4) Black-box KD. The three distillation methods men-tioned above rely on the premise that internal informationof the teacher model is accessible, so they all fall underthe category of white-box distillation(distillation method thatrequires access to internal data of the teacher model duringthe training process). However, many contemporary closed-source large models have inaccessible internal information,and we can only obtain the model’s predictions. The distil-lation pattern where knowledge is transmitted through thepredictions of the teacher model is referred to as black-boxknowledge distillation.

# 5.2 KD for Medium-Size Language Models

With the emergence of the transformer architecture, var-ious medium-size language models based on the trans-former structure (e.g. BERT, GPT-2, etc), have been pro-posed. These language models are trained through twotraining processes: pretraining and finetuning. Specifically,in the pretraining phase, we train the model on a large-scale unlabeled dataset to learn the general features andstructure of language. Subsequently, during the finetuningprocess, we further train the model on labeled data toadapt it to the specific features and requirements of thegiven task. Consequently, unlike previous distillation meth-ods, distillation for these models is categorized into twoclasses: finetuning distillation and pretraining distillation.The student model can receive knowledge transmitted fromthe pretrained teacher during the pretraining period orfrom the teacher fine-tuned for a specific task during thefinetuning period. We will separately introduce these twodistillation paradigms. Additionally, we have created theTable 4 to illustrate the training stage, knowledge sourceand loss function for various medium-size model distillationmethods mentioned below.

# 5.2.1 Finetuning Distillation

Finetuning distillation is primarily aimed at compressingmodels for specific tasks. Generally, teachers in finetun-ing distillation are models that have been fine-tuned forspecific tasks. For example, Distilled BiLSTM [182] is theearliest method to employ knowledge distillation on BERT.It transfers the knowledge of fine-tuned BERT to BiLSTMby learning from logits. Therefore, this is a successful imple-mentation of logit-based knowledge distillation on medium-size models. Subsequently, many feature-based knowledgedistillations [183], [184] have also been implemented onmedium-size models. They distill knowledge in the embed-ding layer, transformer layers, and prediction layer, allow-ing the student model to learn the knowledge mastered


TABLE 4: A summary of various KD methods for BERT. Embed., Attn., Hidden., and Pred. represent knowledge is fromembeddings, attentions, hidden layers, and model’s prediction, repectively.


<table><tr><td>KD Method</td><td>Training stage</td><td>Embed.</td><td>Attn.</td><td>Hidden.</td><td>Pred.</td><td>New Knowledge Source</td></tr><tr><td>Distilled BiLSTM [182]</td><td>Finetuning</td><td></td><td></td><td></td><td>MSE</td><td></td></tr><tr><td>PKD [183]</td><td>Finetuning</td><td></td><td></td><td>MSE</td><td>CE</td><td></td></tr><tr><td>DynaBERT [184]</td><td>Finetuning</td><td>MSE</td><td></td><td>MSE</td><td>CE</td><td></td></tr><tr><td>Metadistil [185]</td><td>Finetuning</td><td></td><td></td><td></td><td>CE</td><td></td></tr><tr><td>AD-KD [186]</td><td>Finetuning</td><td></td><td></td><td></td><td>CE</td><td>Attribution map (MSE)</td></tr><tr><td>AdaBERT [187]</td><td>Finetuning</td><td></td><td></td><td>CE</td><td>CE</td><td>Model efficiency</td></tr><tr><td>MixKD [188]</td><td>Finetuning</td><td></td><td></td><td></td><td>CE</td><td>MixUp data (CE/MSE)</td></tr><tr><td>Meta-KD [189]</td><td>Finetuning</td><td>MSE</td><td>MSE</td><td>MSE</td><td>CE</td><td>Transferrable knowledge (MSE)</td></tr><tr><td>ReAugKD [190]</td><td>Finetuning</td><td></td><td></td><td></td><td>CE</td><td>Similarity matrix (KL)</td></tr><tr><td>DistilBERT [191]</td><td>Pretraining</td><td></td><td></td><td>COS</td><td>CE</td><td></td></tr><tr><td>MiniLM [192]</td><td>Pretraining</td><td></td><td>KL</td><td></td><td></td><td>Value-relation (KL)</td></tr><tr><td>MobileBERT [193]</td><td>Pretraining</td><td></td><td>KL</td><td>MSE</td><td>MSE</td><td></td></tr><tr><td>HomoBERT [194]</td><td>Pretraining</td><td>MSE</td><td>MSE</td><td>MSE</td><td>KL</td><td></td></tr><tr><td>TinyBERT [195]</td><td>Finetuning and pretraining</td><td>MSE</td><td>MSE</td><td>MSE</td><td>CE</td><td></td></tr><tr><td>TED [196]</td><td>Finetuning or Pretraining</td><td></td><td></td><td></td><td>KL</td><td>Filters (MSE)</td></tr></table>

by the teacher model from various aspects. For example,PKD [183] introduced a hidden state loss. It selects a subsetof outputs from the intermediate transformer blocks ofboth the teacher and student for distillation. Additionally,it designed two alignment modes, namely PKD-Skip (thestudent learns from every k layers of the teacher) and PKD-Last (the student learns from the last k layers of the teacher),with experimental evidence demonstrating the superiorityof the former. DynaBERT [184] also takes into account thewidth of the model, and it incorporates the idea of pruning.To be specific, It sets a parameter, the width multiplier$m _ { w } \in ( \bar { 0 } , 1 )$ , and retains the most important $m _ { w }$ attentionheads in the Multi-Head Attention (MHA) layer of thetransformer, as well as the most important $m _ { w }$ neurons inthe Feed-Forward Network (FFN), to initialize the studentmodel DynaBERTw. Then it transfers knowledge from theteacher model to the width-adaptive DynaBERTw throughthe embedding layer, hidden states, and the predictionlayer. Following that, it uniformly selects transformer layersfrom DynaBERTw using the depth multiplier $\mathrm { m _ { d } }$ (similarto PKD-skip) to initialize the student model DynaBERT.Knowledge is then transferred from DynaBERTw to boththe width-adaptive and depth-adaptive DynaBERT usingthe same knowledge source as in the width-adaptive pro-cess. Metadistil [185] points out two common issues ingeneral distillation: the teacher cannot perceive the stu-dent’s abilities, and a strong teacher may not necessarilybe effective in teaching good students. To address theseproblems, it proposes a novel distillation approach: firstdistill a copy ${ \boldsymbol { \mathrm { S } } } ^ { \prime }$ of the student S on training data, then usethe updated ${ \boldsymbol { \mathrm { S } } } ^ { \prime }$ to update the teacher model on quiz data,allowing it to learn to teach. Finally, use the updated teachermodel to distill S on training data. AD-KD [186] focuses onthe importance of each token to the prediction results. Itaims for the student model to understand which tokens theteacher model prioritizes when generating predictions, thuslearning the rationale behind the teacher model’s reasoning.

Some methods [183], [184] mentioned above can beapplied to pretraining distillation from the perspective of

operational feasibility, but Turc et al. [197] has demonstratedthat simple pretraining distillation methods result in signif-icant distillation losses. Therefore, the effectiveness of thestudent models distilled using these methods may not beideal. Besides, some methods [185], [186] have not beenutilized in pretraining distillation. The applicability of thesemethods to pretraining distillation remains to be explored.

Considering the fact that finetuning distillation is tai-lored for specific tasks, many other methods also utilize pro-prietary knowledge sources, enabling students to acquireknowledge these specific tasks need from the teacher modelmore efficiently. So, these methods cannot be applied topretraining distillation.

For instance, AdaBERT [187] employs a search space toenable adaptive changes in the student’s structure. Specifi-cally, the search space consists of multiple layers, with eachlayer comprising input nodes, output nodes, and hiddeninternal nodes that form a directed graph. The edges of thisgraph represent candidate operations selected from a seriesof lightweight operations based on CNN. Considering thesize and efficiency of the student model, AdaBERT incor-porates not only soft and hard targets for distillation butalso includes the normalized parameter size and number offloating-point operations of the student model in the lossfunction. Ultimately, this loss function is used to chooseappropriate CNN-based operations. However, MixKD [188]starts with the dataset and applies MixUp [198] to KD in or-der to address the issue of limited training samples leadingto insufficient knowledge acquisition by the student. It useszero padding to make all sentences the same length, andthen interpolates the word embeddings and labels of twotraining samples to obtain an augmented sample. Then it in-corporates the loss of mixup samples into the loss function.Meta-KD [189] recognizes that when a student is learning inone domain, they may benefit from auxiliary knowledge inother domains. For example, a physics student may findit easier to grasp physics equations under the guidanceof a teacher proficient in both physics and mathematics.Hence, training an ”all-purpose teacher” model for domain-

specific student models becomes essential. More precisely, itconstructs a learnable sub-network using the output of thelast hidden layer for each instance. This sub-network is ca-pable of distinguishing the domain of each instance, makingthe knowledge transferable and not restricted by domainlimitations. During the distillation process, the teacher istasked not only with conveying knowledge encompassedby input embeddings, hidden states, attention matrices, andoutput logits but also with transmitting this transferableknowledge. ReAugKD [190] take the inference phase intoconsider. It uses an external memory derived from rel-evant task-specific knowledge of the teacher to enhancethe effective capacity of the student. In the distillationphase, it adds a linear projection head, which has beenfine-tuned for downstream tasks, on top of the teachermodel’s encoder to generate the teacher embedding andobtains the student embedding from the last transformer.Then it trains with a relational KD loss that minimizesthe divergence between teacher-teacher and teacher-studentembedding distributions. They found that this distillationmethod can effectively enhance the student model’s abilityto retrieve external information. In the inference phase, itconstructs a knowledge base with the teacher’s soft labelsand predictions. Then, it processes the top-k data entriesfrom the knowledge base that are most similar to the studentembedding. The final prediction is obtained by weightingand combining the student’s predictions with these pro-cessed entries from the knowledge base.

Besides, Enhanced KD [199] proposes a new distillationloss function by expanding the loss in a Taylor series,which allows for effective distillation even when the teachermodel is not fine-tuned for a specific task. This approach re-duces a significant amount of training cost and architecture-agnostic.

# 5.2.2 Pretraining Distillation

The primary objective of pretraining distillation is to obtaina pretrained model with fewer parameters and good gen-eralization capabilities. So some of them [191], [193], [194]utilize the loss function employed during the training ofBERT. DistilBERT [191] is the first to introduce pretrainingdistillation for BERT. It transfers the idea of PKD-skip [183](the student learns from every k layers of the teacher) topretraining distillation and employs the cosine similarityloss function to facilitate the transfer of knowledge withinhidden states. MiniLM [192] places the emphasis of distilla-tion on the last transformer layer. It utilizes the self-attentiondistributions and self-attention value-relation (dot-productof the value matrix with itself) from this layer to acquireknowledge and perform distillation. This approach cleverlyallows the student to have more flexible layer numbersand hidden dimensions. Hence, it can straightforwardlydistill the teacher into a teacher assistant [200] with smallerhidden dimensions and then distill the teacher assistantinto a student model with fewer layers, thereby enhancingthe performance of the student model. MobileBERT [193]and HomoBERT [194] put emphasis on model width asDynaBERT [184], but they just alter models’ width whilepreserving their depth because Turc et al. [197] proves thatthe impact of depth on model performance is more signif-icant. MobileBERT adds bottleneck and inverted-bottleneck

to both the teacher and student models to alter the hiddendimensions. However, the practical implementation of thisapproach may disrupt the balance of parameters betweenMulti-Head Attention (MHA) and Feed-Forward Network(FFN). Therefore, the authors address this issue by adoptinga stacked FFN approach. Then it distills knowledge throughthe attention and hidden states of transformer layers. Ho-moBERT utilizes the concept of pruning as DynaBERT. Butit initializes the student with the teacher model so that itcan maintain small discrepancy compared to the teachermodel. Then it derives the distillation loss function usinginput embeddings, hidden states, attention matrices, andoutput logits as the pruning objective. In each iteration, itremoves the least important neurons from the student basedon importance scores and guides the student’s trainingusing the distillation loss. This process is iteratively repeatedthroughout the entire training until the student reaches thetarget size. TinyBERT [195] combines pretraining distillationand finetuning distillation so that TinyBERT can capture thegeneral-domain as well as the task-specific knowledge inBERT. It also distills various knowledge from the embeddinglayer, hidden states and attention matrices of transformerlayers, and the prediction layer. But the ablation studiesshow that finetuning distillation has a more significantimpact than pretraining distillation. TED [196] equips eachlayer with a task-aware filter (a neural network with a task-specific head) to extract knowledge from the hidden repre-sentation of this layer. It has achieved promising results inboth pretraining and finetuning scenarios.

# 5.2.3 Discussion

Finetuning distillation is computational costly becauseswitching to a new task always requires the training of atask-specific teacher. So many finetuning knowledge distil-lation methods [189], [190], [199] are proposed to reducethe computational cost of the finetuning process. But inpretraing distillation, student is distilled from a teacherpretrained on open-domain data and can be efficiently fine-tuned on various downstream tasks, which reduces thecomputational cost associated with distillation for multiplespecific tasks to a certain extent. However, pretraining distil-lation also comes with many new challenges. For example,teacher models have larger capacity and stronger represen-tation capabilities than student models, it is challenging forstudents to produce predictions that match the teacher’s ona large amount of open-domain training data. Therefore, forgeneral methods, the choice between pretraining distillationand finetuning distillation depends on the trade-off wemake between model size and performance.

# 5.3 KD for Large Language Models

Recently, an increasing number of large language mod-els(LLMs) have been developed. However, many of theselarge models are closed-source, which imposes significantlimitations on knowledge distillation for such models. Whilethe student model cannot acquire knowledge from internalinformation, we can still use the teacher model’s responses,the remaining source of knowledge, to transfer informationto the student model. Depending on whether the source ofknowledge for the student model is solely the answers pro-vided by the teacher model, distillation for large language

models can be categorized into black-box distillation andwhite-box distillation.

# 5.3.1 Black-box Distillation

Even though conventional distillation methods may nolonger apply, some unique properties of LLMs allow usto find a breakthrough. Researchers have found that whenthe models’ parameter is large enough, they exhibits sur-prising emergent abilities, enabling it to tackle intricatetasks. Many black-box distillation methods leverage thatabilities, and there are typically three methods commonlyin use: Instruction-Following, Chain-of-Thought (CoT) andIn-Context Learning.

# 1) Instruction-Following

Instruction-following capability means that the LLMscan generate corresponding outputs based on a specificinstruction (directing the model on what task to accomplish)and the input (data required to fulfill that instruction).Due to the fact that black-box distillation can only transferknowledge through datasets, it necessitates a sufficientlycomprehensive dataset. Therefore, the common effort inthis method [201], [202], [203], [204] involves constructing alarge dataset (comprising instructions, inputs, and outputs)to enable the student models to learn as much as possiblefrom the teacher models. Specifically, SELF-INSTRUCT [201]employs a self-distillation approach, where the model servesboth as the teacher and the student. It starts by obtaininga manually curated small-scale task pool, where each taskconsists of an instruction and a corresponding input-outputpair. Subsequently, it selects a subset of instructions from thetask pool as in-context examples for the model to generatenew instructions and matching inputs and outputs for them.Finally, it filters out data with excessive redundancy orcontent that cannot be handled by language models, placingthe qualified data back into the task pool. This iterativeprocess continues to generate an extensive dataset for fine-tuning the student model. This has become a paradigm forinstruction-following distillation, and the 13B open-sourcemodels Alpaca [205], Vicuna [206] and GPT4All [207] weretrained with some adjustments based on this paradigm.Also, following this idea, LLaMA-GPT4 [202] and LaMini-LM [203] construct their respective instruction sets andfine-tune smaller models. Compared to SELF-INSTRUCT,their breakthroughs are as follows: LLaMA-GPT4 generatesa 52K instruction-following dataset in both English andChinese using GPT-4, and fine-tunes two student models,LLaMA-GPT4 and LLaMA-GPT4-CN. Additionally, it trainsa reward model specifically for evaluating the quality ofmodel responses. LaMni-LM enriches the types of modelsused for generating instructions and the topics of instruc-tions, constructing a massive dataset of 2.58M for finetuningsmaller-parameter student models, which achieves goodresults. However, in the methods mentioned above, thestudent model is not involved in the selection of the dataset,so the teacher model cannot receive timely feedback fromthe student model during the dataset generation process. Inresponse to this issue, Lion [204] adopts adversarial knowl-edge distillation, where the student model not only learnsfrom the teacher model’s responses but is also evaluatedby a referee to assess its difference compared to the teachermodel. This helps to identify ”hard” instructions where the

student model’s performance falls short, thus generatingnew ”hard” instructions so that teacher models can achievefeedback in the learning process. PERsD [208] evaluates thestudent’s attempt with unit test cases and gets executionfeedback. Then it prompts the teacher model to refine thestudent’s attempt so that the student can be trained onpersonalized data.

Some work focuses on task-specific instruction-following distillation. For instance, UniversalNER [209]conducts in-depth research on Named Entity Recognition(NER) tasks. So, unlike the methods mentioned above thatincrease the diversity of instructions, its emphasis is onenhancing the diversity of inputs to improve the model’sgeneralization across multiple domains. To be specific, itdirectly samples inputs from a large corpus across diversedomains, and then uses a LLM to generate outputs. Afterobtaining the data, it trains the student model using aconversation-style tuning format, enabling it to identifyentities of each entity type contained in the input text.

Furthermore, this approach of using large languagemodels to construct reinforced datasets for finetuning stu-dent models is not unique to instruction-following butrather a common method of black-box distillation.

# 2) Chain-of-Thought

Chain-of-Thought capability refers to the ability of alarge language model to provide better answers to questionsbased on the rationale within the given prompts. The typicalparadigm of CoT [210], [211], [212], [213] distillation utilizeslarge models to generate reinforced datasets containing ra-tionales, which are then used to fine-tune the student model.Hence, the issues of interest revolve around how to generatehigh-quality rationales for training [210], [214], [215], [216],[217], [218], [219], [220], [221], [222], [223] and how to en-sure that students effectively leverage these rationales [210],[212], [215], [216], [217], [223], [224].

Li et al. [210] systematically explores three explanationgeneration approaches from LLMs and three multi-tasklearning with explanations methods. Finally it finds thatCROP (Chain of Thought with Rationalization Promptingbackup) and MT-CoT (Multi-task Learning with Chain ofThought) are outstanding methods. In detail, CROP refers toa process where, for a dataset containing questions and an-swers, the teacher model first produces an explanation andan answer based on the question. If the answer is correct,the explanation is retained. If the answer is incorrect, theteacher model generates an explanation based on the ques-tion and the correct answer. Ultimately, a dataset is obtainedwith questions, explanations, and answers for finetuningthe student model. MT-CoT refers to a training process forthe student model with two tasks. The model is not onlyrequired to learn predicting answers but also to provide ex-planations. Moreover, in the task of providing explanations,the model needs to arrive at the correct answer throughthe reasoning steps it takes. Further, Distilling Step-by-Step[212] demonstrates that good results can be achieved evenwhen the original dataset only consists of questions with-out answers. Fine-tune-CoT [214] applies existing zero-shotCoT prompting to generate rationales from large teachermodels, and uses them to fine-tune smaller student models.It also proposes diverse reasoning to augment the trainingdata for student models so that student models can have

better performance. Besides, SCoTD [220] and MCC-KDc[221] also conducts in-depth explorations on the diversity ofrationales. Fu et al. [225] found that it is indeed possible totransfer the student model’s capabilities from general tasksto tasks specific by employing CoT distillation. SOCRATICCoT [215] decomposes a question into several sub-questionsto guide the generation of rationales. It starts by selectinga subset of data from the dataset, manually decomposingquestions, and providing answers for each sub-question.These serve as examples given to the LLM to generate sub-questions and answers for the remaining data. The resultingdataset is reinforced by filtering based on the correctness ofthe final results. Two student models are then trained usingthis dataset, one for questioning and one for answeringquestions. SCOTT [216] takes into account two issues inCoT. Firstly, the rationale generated by the teacher modelmay not match the answer or be meaningful. Secondly, thestudent model may struggle to connect rationale and answerduring learning. To address these challenges, SCOTT em-ploys contrastive decoding during the rationale generationprocess to make the model pay more attention to the answer.This requires the teacher model’s decoding process to beadjustable. In the training process of the student model,SCOTT introduces counterfactual rationales to guide thestudent in obtaining different answers, thereby establishinga closer relationship between rationale and answer. KARD[217] addresses the issue of limited memory capabilitiesin small models by retrieving information from externalknowledge bases. Program Distillation [218] and PaD [219]both leverage programs as rationales and have achievedpromising results on math word problems. DOCTOR [222]utilizes a teacher model to generate question-answer-stylerationales containing commonsense knowledge, and thenfilters and selects high-quality multi-hop reasoning for train-ing students. Wang et al. [223] build an interactive multi-round learning paradigm, where the student first providesits learning status to the teacher LLM who then can pro-vide customized rationales as the feedback to the student.They also exploit the reasoning potential of smaller LM byeliciting it to take self-reflection on the mistakes.

# 3) In-Context Learning

In-context learning (ICL) is also a manifestation of theemergent capabilities of large models, referring to the ca-pacity of large models to generate correct outputs for newinputs based on some input-label examples without up-dating model parameters. Based on it, In-context LearningDistillation [226] utilizes two few-shot learning paradigms,namely Meta In-context Tuning (Meta-ICT) and MultitaskIn-context Tuning (Multitask-ICT), to transfer the in-contextlearning capabilities of teacher models to student modelsby distillation. In Meta-ICT, it enables the student modelto adapt to unseen tasks through in-context learning andassistance from the teacher. But in Multitask-ICT, it treatsall target tasks as training tasks and directly employs ex-amples from target tasks in in-context learning distillation.The results demonstrate that multi-task in-context tuning ismore effective, although it comes with higher computationalcosts. LLM-R [227] initially trains a reward model based onLLM feedback to evaluate the quality of candidate exam-ples, followed by knowledge distillation to train a retrieverthat can identify high-quality in-context examples for LLMs.

# 4) Others

In addition to the three paradigms mentioned above,there are other methods that generate specific reinforcementdatasets to enable the student model to acquire specificcapabilities. For instance, Symbolic Knowledge Distillation[228] utilizes a LLM to gather data and filter it, thereby ob-taining high-quality Commonsense Knowledge Graphs fortraining a Commonsense Model. DISCO [229] uses a LLM toobtain counterfactual data and employs a large teacher NLImodel for filtering, thus obtaining a high-quality datasetto improve students’ abilities in natural language inference(NLI) tasks. PubMedBERT [230] conducts a case study onadverse drug event (ADE) extraction and proposes a novelframework that simultaneously handles adverse event (AE)entity extraction and ADE relation extraction to reduce com-putational requirements. Promptmix [231] utilizes LLMsto mix and relabel text data for classification problems inproportion, aiming to obtain a stronger dataset for training.

However, Gudibande [232] demonstrates that continu-ally increasing imitation training data can lead to the modelsimply imitating without understanding, thus enhancingthe capabilities of the base model is also an indispensableaspect of black-box distillation.

# 5.3.2 White-box Distillation

Compared to black-box distillation, the work on white-box distillation is relatively limited, but there is still someexploration. For example, MINILLM [233] and GKD [234]both focus on the loss function and they find that forwardKL divergence overestimates the void regions of the teacherdistribution in language generation tasks when the studentmodel distribution is insufficiently expressive to cover allthe modes of teacher distribution. But reverse KL divergencefocuses on the major modes, allowing the student to learnthe main part of the teacher’s distribution. Furthermore, itdoesn’t force the student to exactly match the teacher’s dis-tribution but aims to leverage the information provided bythe teacher to assist in the student’s training. So MINILLMsamples from the student distribution and uses the PolicyGradient Theorem to calculate the reverse KL divergence.Also due to the high variance and reward hacking policygradient suffers from, it comes up with the single-stepregularization, teacher-mixed sampling and length normal-ization to solve these problems. Similar to MINILLM, GKDutilizes reverse KLD and Jensen-Shannon divergence (JSD)to enhance the student’s expressive capacity. But it uses on-policy KD to alleviate the distribution mismatch betweentraining and evaluation, which involves sampling fromthe student distribution without backpropagating throughstudent’s sampling process—something that MINILLM re-quires. It’s proved that this gradient handling approach isrelatively simple yet effective. Padmanabhan et al. [235]generate a transfer set by prompting a language model togenerate continuations from the entity definition and thenupdate the model parameters so that the distribution ofthe student matches the distribution of the teacher on thetransfer set. TSLD [236] utilizes logit distillation to reformintermediate representations and applies token-wise logitscaling, reducing the errors introduced when QAT is appliedto generative language models. MiniMA [237] finds that theoptimal distillation effect occurs when the student model

is approximately $4 0 \%$ of the size of the teacher model’sparameters. It utilizes LLaMA2-7B for structured pruningand logit-based knowledge distillation to train a 3B MiniMAmodel.

Due to the limitations imposed by the closed-sourcenature of large language models, white-box distillationhas faced constraints. However, with the emergence ofincreasingly diverse open-source large language models(e.g. Alpaca, Vicuna), white-box distillation holds significantpromise for the future.

# 6 COMPACT ARCHITECTURE DESIGN

Compact architecture design is a philosophy that pursuesefficiency and streamlining, and it aims to achieve a signifi-cant increase in model efficiency by optimizing the networkstructure and algorithms while reducing the consumptionof computational resources and memory usage. Specifically,it can be divided into two levels of research: micro andmacro. This section will focus on optimizing the attentioncomputation and the Transformer architecture design. Sincethe Transformer layer is currently the main component ofthe LLM, and it makes no difference for large and medium-size models, so we will not specifically categorize methodsby model size here.

# 6.1 Efficient Attention

The standard self-attention mechanism of the Transformerhas a time and space complexity of $O ( N ^ { 2 } )$ for sequencelength $N _ { \cdot }$ , which significantly limits its further expansion invarious fields and prevents it from handling long-sequenceproblems. To solve this problem, many works have emergedto improve attention, many of which have focused on im-proving computational and memory efficiency. We refer tothese works as Efficient Attention. Based on the startingpoint and method characteristics, we divide these worksinto three categories: Sparse Attention, Linear Approxi-mate Attention, and Flash Attention. There are also someunique works, such as Transformer-XL [238], which do notimprove within the attention operator and, therefore, willnot be discussed here.

# 6.1.1 Sparse Attention

The Sparse Attention approaches [239], [240], [241], [242],[243], [244], [245], [246], [247], [248], [249], [250] allow eachtoken to attend only locally or predominantly relevant itemsto implement the sparse attention pattern, thus reducingcomputational and memory complexity. Based on the char-acteristics of these methods, we categorize them into stride-based, window-based, and data-based methods.

# 1) Stride-based Methods

The stride-based methods [239], [240], [241] reduce com-putational complexity by having each token attend to sev-eral preceding tokens of length stride to achieve sparseattention patterns.

[239] is an earlier work. It offered two ways to de-compose attention: strided (Fig. 6 (b)) and fixed attentionpatterns. These ways allowed each query to attend only topreset positions, reducing the complexity of self-attention to√$\overset { \cdot } { O } ( N \overset { \cdot } { \sqrt { N } } )$ . However, this method has limited applicability

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-11/0a775777-6cdd-4dbe-a304-295f64b9a1ad/49eadbf4994603b10c7ad999a4683a6d09bbdd3941e32caa23338a977a60edd7.jpg)



(a) Full attention


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-11/0a775777-6cdd-4dbe-a304-295f64b9a1ad/6fe73685ce2b4720f037643cd6393d69ceae4294c1a2028f122fefc9ee7ca9ce.jpg)



(b) Sparse attention (strided)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-11/0a775777-6cdd-4dbe-a304-295f64b9a1ad/dd16d24a664e729d66b2b0363cee250117fbb9ac0d8e35bac89f8b3d53a7acee.jpg)



(c) Window attention


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-11/0a775777-6cdd-4dbe-a304-295f64b9a1ad/22feeeff228be8eb7e2b2657438a94bbad31a981c90d498ef8a6032c8c2c73f0.jpg)



(d) Global attention



Fig. 6: Comparing the sparse attention patterns. (a) full self-attention, (b) strided attention, (c) window attention, (d)global attention.


unless we can design appropriate sparse attention kernelsfor various scenarios. By observing the distribution of at-tention across different heads in a standard Transformer,[240] found that not all heads attend to the entire context(some heads only focus on the nearest tokens) and proposedlearning dynamic attention spans for each head to reducecomputational and storage costs. However, both previousworks can only attend to past consecutive spans of tokens.To solve this problem, [241] introduces $\alpha \mathrm { . }$ -entmax to replacesoftmax, allowing low-rated words to receive precisely zeroweight, thus enabling more flexible sparse attention.

# 2) Window-based Methods

Unlike the above approaches, the window-based meth-ods divide the input into individual localized windows sothat each token only attends to items inside the window(Fig. 6 (c)), thus reducing the computational complexity.

An early method [243] achieves a complexity of $\begin{array} { r } { \dot { O ( \frac { N ^ { 2 } } { n } ) } } \end{array}$by dividing the $Q , K ,$ , and $V$ matrices into $n$ blocks (paddingis used if not divisible) and calculating attention within eachblock by shifting one position. This method is more straight-forward to implement than Sparse Transformer. However, itis worth noting that $n$ is typically set to 2 or 3 in practiceto maintain performance, which results in poor actual accel-eration. To solve this problem, [244] achieves $O ( N ( k + m ) )$complexity by using a dilated sliding window to increasethe receptive field without increasing the computation andimproving the performance by adding ”Global Attention”(Fig. 6 (d)) to the pre-selected input positions. Here, $k$represents the size of the sliding window, and $m$ repre-sents the number of pre-selected positions. However, itsextended implementation requires efficient banded matrixmultiplication support, and using the naive CUDA kernelscan only have the running speed as standard self-attention.Therefore, in practice, it only has a consistent memory foot-

print with theoretical complexity, but there is a gap betweenactual running speed and theoretical complexity. Similarly,[246] established a sparse attention pattern consisting ofthree main components:

Global attention: A global token set $g$ where tokenswithin the set attend to the entire sequence, and alltokens in the sequence attend to set $g$ .

Local attention: All tokens attend to a set $w$ of sur-rounding windows around themselves.

Random attention: All tokens attend to a randomtoken set $r$ .

It is worth noting that there is also a gap between BigBird’sactual running speed and theoretical complexity, similar to[244].

In addition to the methods described above, some meth-ods [242], [249] let each token attend directly to close andindirectly to distant locations. This method is very similarto the window-based methods introduced above but witha different indirect way to implement ”Global Attention.”

Therefore, we present them here.

[242] proposes a new architecture, BP-transformer(BPT), based on the prior observation that elements closertogether have higher attention scores, while elements fur-ther away have lower attention scores. It treats attentioncalculation as a graph neural network and partitions theinput sequence into different multi-scale spaces throughbinary partitioning (BP), constructing a binary tree-basedattention pattern. Each leaf node represents a token, andeach token focuses on different scale nodes based on thetarget distance, thereby reducing the complexity of attentionto $\bar { O } ( k N l o g ( N / k ) )$ where $k$ is a hyperparameter controllingthe density of attention. The core idea of [249] is very similarto BPT in that it enables each token to attend to all otheritems directly or indirectly. The difference is that it treats theattention mechanism as a conditional expectation problem.

# 3) Data-based Methods

Unlike the above methods that need to design sparsepatterns manually, data-based methods [245], [247], [248],[250] make each token automatically and quickly find themost relevant items to compute attention using appropriatealgorithms. The most significant advantage of these meth-ods is data-awareness, which effectively avoids the disad-vantage of having to re-design the sparse patterns manuallyin the case of different tasks and data, and it isn’t easy toobtain the optimal solution.

Reformer [245] achieves efficient sparse attention com-putation by using locally sensitive hashing to find similarvectors quickly, reducing the complexity to $O ( N l o g ( N ) )$ .At the same time, Reformer also uses techniques such as re-versibility layers and chunking in FFN layers to significantlyreduce memory usage during training. However, this trade-off may also slow down the training speed. In addition,to avoid hash errors, Reformer requires multiple rounds ofhashing, weakening its final efficiency benefits. Similarly toReformer, [250] views self-attention as a routing problem.Specifically, it is based on k-means clustering, which allowsqueries and keys to cluster on the same set of cluster center-of-mass vectors by letting the model learn to select sparseclusters of word examples. So that each query $Q _ { i }$ attendsonly to the keys that belong to the same cluster as it does.

To ensure performance, it sets the number of clusters to $\sqrt { N }$ ,which reduces the attention complexity to $O ( N { \sqrt { N } } )$ .

Other works related to sparse attention based on inputare SAC [248] and SSA [247]. Among them, SAC regardsthe input as a graph and uses an LSTM edge predictor tolearn the edges between tokens. The nodes in the graphrepresent tokens, and the edges represent attention rela-tions. It also uses reinforcement learning to train this edgepredictor. However, LSTM has limitations, such as a lackof parallelism and a limited ability to express long-termdependencies. There may be better methods available forbuilding an edge predictor. On the other hand, SSA is basedon the differentiable sorting of internal representations andintroduces a meta-sorting network that can learn to gen-erate potential orderings on sequences. It allows us to useonly local windows for quasi-global attention after a givenordering sequence, improving the memory efficiency of theattention module.

# 6.1.2 Linear Approximate Attention

The standard attention can be represented as:

$$
\operatorname {A t t e n t i o n} (Q, K, V) = \operatorname {s o f t m a x} \left(Q K ^ {\mathcal {T}}\right) V \tag {28}
$$

Since $Q K ^ { T }$ is quadratic in sequence length and memorycomplexity, this severely limits applying attention to longsequence scenarios. Therefore, several methods devoted tolinearized attention computation have been proposed toaddress this dilemma. Based on the characteristics of thesemethods, we categorize them into associativity-based andlow-rank-based methods.

# 1) Associativity Based Methods

The natural idea is that if we can calculate $K ^ { T } V$ firstutilizing the associativity of matrix multiplication, we canachieve linear complexity in attention computation. How-ever, due to the presence of softmax, we cannot easilyimplement this. For each row $i$ in the attention result, wecan equivalently represent it as:

$$
\operatorname {A t t e n t i o n} (Q, K, V) _ {i} = \frac {\sum_ {j = 1} ^ {n} \sin \left(q _ {i} , k _ {j}\right) v _ {j}}{\sum_ {j = 1} ^ {n} \sin \left(q _ {i} , k _ {j}\right)} \tag {29}
$$

Where $s i m ( q _ { i } , k _ { j } ) = e ^ { q _ { i } k _ { j } ^ { T } }$ , it is actually a weighted averageof $v _ { j }$ with weights given by $e ^ { q _ { i } k _ { j } ^ { T } }$ . A natural thought is thatif we can find two functions $\phi _ { 1 } ( x )$ and $\phi _ { 2 } ( x )$ such that:

$$
\sin \left(q _ {i}, k _ {j}\right) = \phi_ {1} \left(q _ {i}\right) \phi_ {2} \left(k _ {j}\right) ^ {T} \tag {30}
$$

and satisfy $s i m ( q _ { i } , k _ { j } ) > = 0$ all the time, and also satisfy:

$$
\left(\phi_ {1} \left(q _ {i}\right) \phi_ {2} \left(k _ {j}\right) ^ {T}\right) v _ {j} = \phi_ {1} \left(q _ {i}\right) \left(\phi_ {2} \left(k _ {j}\right) ^ {T} v _ {j}\right) \tag {31}
$$

Then, we can achieve linear attention. Building on this idea,many different approaches to linear attention have beenproposed [251], [252], [253], [254], [255], [256].

Specifically, [251] achieves this by constructing:

$$
\phi_ {1} (x) = \phi_ {2} (x) = e l u (x) + 1 = \left\{ \begin{array}{l l} 1 + x & x \geqslant 0 \\ e ^ {x} & x <   0 \end{array} \right. \tag {32}
$$

Performer [252] also achieves linear attention through akernel function method. It proposes a $\mathrm { F A V O R } +$ methodthat cleverly uses random projection to project the inputfeatures orthogonally. Without relying on any prior and

without loss of accuracy, it successfully realizes the linearattention. Specifically, by taking $\phi$ of the following formfor functions $f _ { 1 } , . . . , f _ { l } : \mathbb { R }  \mathbb { R } ,$ function $g : \mathbb { R } ^ { \breve { d } } \to \mathbb { R }$and deterministic vectors $\omega _ { i }$ or $\omega _ { 1 } , . . . , \omega _ { m } \stackrel { \mathrm { i i d } } { \sim } \mathcal { D }$ for somedistribution $\mathcal { D } \in \mathcal { P } ( \mathbb { R } ^ { d } )$ :

$$
\phi (\mathbf {x}) = \frac {h (\mathbf {x})}{\sqrt {m}} \left(f _ {1} \left(\omega_ {1} ^ {\top} \mathbf {x}\right), \dots , f _ {1} \left(\omega_ {m} ^ {\top} \mathbf {x}\right), \dots , f _ {l} \left(\omega_ {1} ^ {\top} \mathbf {x}\right), \dots , f _ {l} \left(\omega_ {m} ^ {\top} \mathbf {x}\right)\right) \tag {33}
$$

To better describe $f _ { i } , h$ and $\omega _ { i }$ in $\phi ,$ for the element $A ( i , j ) =$$e x p ( q _ { i } k _ { j } ^ { T } )$ in the ith row and $j$ th column of the originalattention matrix $A$ , we give it a generalized definition:

$$
\operatorname {S M} (\mathbf {x}, \mathbf {y}) \stackrel {\text {d e f}} {=} \exp \left(\mathbf {x} ^ {\top} \mathbf {y}\right) \tag {34}
$$

In fact, as early as [257] there was an approximate expres-sion for $\operatorname { S M } ( \mathbf { x } , \mathbf { y } )$ with $\begin{array} { r } { h ( x ) = e x p ( \frac { | | x | | ^ { 2 } } { 2 } ) } \end{array}$ , l = 2, f1 = sin,$f _ { 2 } ~ = ~ c o s$ . Since the previous methods appear to have sinand cos trigonometric functions and instabilities such asnegative numbers may appear in the computed results,Performer proposes another, more stable, approximation:

$$
\begin{array}{l} \operatorname {S M} (\mathbf {x}, \mathbf {y}) = \mathbb {E} _ {\omega \sim \mathcal {N} (0, \mathbf {I} _ {d})} \left[ \exp \left(\omega^ {\top} \mathbf {x} - \frac {\| \mathbf {x} \| ^ {2}}{2}\right) \right. \tag {35} \\ \left. \exp \left(\omega^ {\top} \mathbf {y} - \frac {\| \mathbf {y} \| ^ {2}}{2}\right) \right] = \Lambda \mathbb {E} _ {\omega \sim \mathcal {N} (0, \mathbf {I} _ {d})} \cosh (\omega^ {\top} \mathbf {z}) \\ \end{array}
$$

where $\begin{array} { r } { \Lambda = \exp ( - \frac { \| \mathbf x \| ^ { 2 } + \| \mathbf y \| ^ { 2 } } { 2 } ) , \mathbf x , \mathbf y \in \mathbb R ^ { d } , \mathbf z = \mathbf x + \mathbf y } \end{array}$ and coshis hyperbolic cosine. This is equivalent to making:

$$
h (x) = \exp \left(- \frac {\left| | x | \right| ^ {2}}{2}\right), l = 2, f _ {1} (x) = \exp (x), f _ {2} (x) = \exp (- x) \tag {36}
$$

However, to ensure accuracy, the number of random sam-ples $m$ is usually larger than the feature dimension ${ \mathrm { d } } ,$which means that when dealing with short sequences, thePerformer may not perform as well as the standard Trans-former. Only when the sequence is relatively long can itsadvantage be fully leveraged. Similarly, [256] achieves linearapproximate attention through a double softmax approach:

$$
\operatorname {A t t e n t i o n} (Q, K, V) \approx \operatorname {s o f t m a x} _ {1} (Q) \operatorname {s o f t m a x} _ {2} (K) V \tag {37}
$$

Where, sof tmax1, sof tmax2 refer to softmax operations inthe first $( N )$ and second $( d )$ dimension, respectively. How-ever, directly softmaxing $Q , K ^ { T }$ separately, i.e., withoutsimilarity (inner product) computation, gives the impres-sion of running counter to the attention mechanism. [253]builds on this by first considering $Q , K$ as $n$ d-dimensionalvectors, respectively, and then clustering them into matricesconsisting of $m$ cluster centers $\tilde { \boldsymbol { Q } } , \tilde { \pmb { K } } \in \mathbb { R } ^ { m \times d }$ . In addition,it inserts a matrix $M \in \mathbb { R } ^ { m \times m }$ in the middle such that thefinal attention computation can be represented as

$$
\begin{array}{l} \operatorname {A t t e m t i o n} (Q, K, V) \approx \operatorname {s o f t m a x} \left(Q \tilde {K} ^ {\top}\right) \\ \left(\operatorname {s o f t m a x} \left(\tilde {Q} \tilde {K} ^ {\top}\right)\right) ^ {- 1} \operatorname {s o f t m a x} \left(\tilde {Q} K ^ {\top}\right) V \end{array} \tag {38}
$$

which is closer to the standard attention.

Recently, HyperAttention [255] simplified the existingalgorithm based on Kernel Density Estimation (KDE), iden-tified the main entries in the attention matrix throughHamming-ordered locally sensitive hashing, and proposeda simple linear time attention approximation algorithm. This

algorithm can achieve a wide range of linear approximationattentions while ensuring the spectral properties of atten-tion and supporting causal masks. It is worth noting thatthe acceleration effect of HyperAttention can be tens oftimes different in two cases of using causal masks and notusing causal masks. At the same time, if HyperAttentioncompletely replaces all layers of attention, the model per-formance will be significantly reduced, so this method stillneeds to balance speed and performance.

# 2) Low-rank Based Methods

Other methods [258], [259] to achieve linear attentionare through the utilization of low-rank property. Linformer[258] observed that the normalized cumulative singularvalues of the attention matrices in the Transformer modelexhibit low-rank properties across multiple tasks. Based onthis observation, Linformer preserves the original Scaled-Dot Attention formulation but projects $K$ and $V$ with twomatrices $\boldsymbol { E } , \boldsymbol { F } ~ \in ~ \mathbb { R } ^ { m \times n }$ before computing attention, en-abling linear approximate attention computation, formallyexpressed as:

$$
\operatorname {A t t e n t i o n} (Q, K, V) \approx \operatorname {s o f t m a x} (Q (E K) ^ {T}) F V
$$

In order to maintain its performance, it’s essential to setthe value of $m$ high enough. However, this can resultin Linformer processing short sequences relatively slowly.Consequently, there might be a significant difference be-tween the theoretical complexity and the practical usage ofLinformer.

Recently, Transformer-VQ has adopted a unique per-spective, performing Vector Quantization (VQ) on the keymatrix $K$ in attention calculation. This is achieved by mak-ing each vector in $K$ closest to the vector in $C ,$ where $C$is the training parameter and the VQ codebook. VQ canbe mathematically represented as: $\widehat { K } \ = \ V Q ( K , C ) , K \ \in$$R ^ { n \times d _ { k } }$ , $C \in { \cal R } ^ { c \times d _ { k } }$ Since each vector in $\widehat { K }$ is from $C$ , wecan first calculate $Q C ^ { T } .$ , which is linear due to the fixed sizeof $C$ . Transformer-VQ cleverly constructs a $\triangle \in \{ 0 , 1 \} ^ { n \times c }$such that:

$$
e x p \left(Q K ^ {T}\right) V = e x p \left(Q C ^ {T} \triangle^ {T}\right) V = e x p \left(Q C ^ {T}\right) \left(\triangle^ {T} V\right) \tag {39}
$$

The computational complexity of this calculation is$O ( n c d _ { k } + n c d _ { v } + n c d _ { v } ) = \bar { O } ( n )$ , achieving linear attention.Moreover, this method can naturally be applied to autore-gressive tasks, making it a promising approach.

Unlike the previous approximate attention methods,FlashAttention [260] focuses its improvements on reducingthe memory access overhead with great success. It achievesacceleration of training and inference while reducing thememory footprint of attention. More importantly, it is not anapproximate attention method, meaning its computationalresults are precisely equal to the standard attention results.

At its core is tiling, which is the chunked transfer ofmatrices involved in computation to shared memory to im-prove overall read and write speed. For example, as shownin Fig. 7 , suppose we now need to compute the upper halfof the result of multiplying a matrix $A$ with a matrix $B , C _ { 0 }$ .For standard matrix multiplication there is $C _ { 0 } = ( A _ { 0 } B _ { 0 } +$$\ldots + A _ { 0 } B _ { 3 } ) c o n c a t ( A _ { 1 } B _ { 0 } + \ldots + A _ { 1 } B _ { 3 } )$ . And for Tiling matrixmultiplication, $C _ { 0 } = ( A _ { 0 } B _ { 0 } + A _ { 0 } B _ { 1 } ) c o n c a t ( A _ { 1 } B _ { 2 } + A _ { 1 } B _ { 3 } )$reduces the memory access to half of the standard matrixmultiplication.

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-11/0a775777-6cdd-4dbe-a304-295f64b9a1ad/d590c98e8a370f0d60b7c697a812acf686b772080e819edfc29814e37a9a56bd.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-11/0a775777-6cdd-4dbe-a304-295f64b9a1ad/8dca710121b405ecde6b1a9581b1cf3e5f58944304b539905d5db8563b3fae75.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-11/0a775777-6cdd-4dbe-a304-295f64b9a1ad/2bea110a138d549a8b10949127dc3213807c214027806a9907b8c34b0ab0f91f.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-11/0a775777-6cdd-4dbe-a304-295f64b9a1ad/e28844dc60815ab29341f74c3755716af74885a2b500143968e989608c822f08.jpg)



Fig. 7: (a):Standard matrix multiplication, (b):Tiling matrixmultiplication.


In fact, in the Naive version of matrix multiplication,only one row and one column of the two matrices areread from memory each time for computation. The memoryaccess efficiency is very low, and the two examples hereread the same number of elements from the matrices A andB, respectively, just for comparability. For matrix multipli-cation, we can use Tiling directly by chunking. Still, thereare softmax operations in attention, and the denominator ofsoftmax contains the summation term associated with all theelements, so the real difficulty in applying tilting to attentionlies in the chunking of softmax.

For standard attention, we usually use the numericallysafe softmax:

$$
\operatorname {S o f t m a x} \left(x _ {i}\right) = \frac {e ^ {x _ {i} - m}}{\sum_ {j = 1} ^ {N} e ^ {x _ {j} - m}} \tag {40}
$$

where $m$ denotes the maximum of all $x _ { i }$ . To get the finalresult, we need three rounds of iterations:

• Iterate over all $x _ { i }$ to find the maximum value $m$

Iterate through all $x _ { i }$ to find the $\begin{array} { r } { s u m = \sum _ { j = 1 } ^ { N } e ^ { x _ { j } - m } } \end{array}$ = PN exj−m

• Calculating each $S o f t m a x ( x _ { i } )$

Since each round of iteration depends on the results ofthe previous iterations, the computation cannot be doneindependently in chunks. One existing method is to definea single sequence $\begin{array} { r } { \boldsymbol { l } ^ { ' } : \boldsymbol { l } _ { i } ^ { ' } = \sum _ { \mathrm { j = 1 } } ^ { \mathrm { i } } \mathrm { e } ^ { \mathrm { x } _ { \mathrm { j } } - \mathrm { m } _ { \mathrm { i } } } } \end{array}$ , thus having:

$$
\begin{array}{l} l _ {i} ^ {\prime} = \sum_ {j = 1} ^ {i} e ^ {x _ {j} - m _ {i}} = \left(\sum_ {j = 1} ^ {i - 1} e ^ {x _ {j} - m _ {i}}\right) e ^ {m _ {i - 1} - m _ {i}} + e ^ {x _ {i} - m _ {i}} \tag {41} \\ = l _ {i - 1} ^ {\prime} e ^ {m _ {i - 1} - m _ {i}} + e ^ {x _ {i} - m _ {i}} \\ \end{array}
$$

It’s a matter of cobbling together a $\begin{array} { r } { \sum _ { j = 1 } ^ { i - 1 } e ^ { x _ { j } - m _ { i } } } \end{array}$ out andreplacing it with an incremental computation of $l _ { i - 1 } ^ { ' } ,$ andit’s clear that, after we get to this point, our sequences can

be computed in the same round of iterations as $l ^ { ' }$ and $m ,$and in the end, $l _ { n } ^ { ' }$ will be equivalent to $l ,$ and in that waywe’ll be able to reduce the three rounds of iterations to tworounds of iterations. However, the two-step iteration is stillcoupled and cannot be chunked for separate computations.Inspired by the previous derivations, FlashAttention derivesmethods to obtain the final O-matrix after one round ofiterations. A row in matrix $O$ is a weighted summation of Vand Softmax results, which can then be expressed as:

$$
o _ {i} \leftarrow \sum_ {j = 1} ^ {N} \left(\frac {e ^ {x _ {j} - m _ {N}}}{l _ {N}} V [ j,: ]\right) \tag {42}
$$

Using and the same trick, introduce a sequence of $o ^ { ' }$ aloneand let it participate in the computation using the local $m _ { i }$and $l _ { i } ^ { ' }$ :

$$
o _ {i} ^ {\prime} \leftarrow \sum_ {j = 1} ^ {i} \left(\frac {e ^ {x _ {j} - m _ {i}}}{l _ {i} ^ {\prime}} V [ j,: ]\right) \tag {43}
$$

It is easy to see that for $N , \ o _ { i }$ is equal to $o _ { N } ^ { ' } ,$ and theproblem translates into figuring out how to cobble togethera Pi−1j=1 $\begin{array} { r } { \sum _ { j = 1 } ^ { i - 1 } \left( \frac { e ^ { x _ { j } - m _ { i - 1 } } } { l _ { i - 1 } ^ { \prime } } V [ j , : ] \right) ^ { \setminus } } \end{array}$  exj −mi−1′ V [j, :] out of the formula replacing itwith $o _ { i - 1 }$ :

$$
\begin{array}{l} o _ {i} ^ {\prime} = \sum_ {j = 1} ^ {i} \left(\frac {e ^ {x _ {j} - m _ {i}}}{l _ {i} ^ {\prime}} V [ j,: ]\right) \\ = \left(\sum_ {j = 1} ^ {i - 1} \frac {e ^ {x _ {j} - m _ {i - 1}}}{l _ {i - 1} ^ {\prime}} V [ j,: ]\right) \frac {e ^ {m _ {i - 1}}}{e ^ {m _ {i}}} \frac {l _ {i - 1} ^ {\prime}}{l _ {i} ^ {\prime}} + \frac {e ^ {x _ {i} - m _ {i}}}{l _ {i} ^ {\prime}} V [ i,: ] \\ = o _ {i - 1} ^ {\prime} \frac {l _ {i - 1} ^ {\prime} e ^ {m _ {i - 1} - m _ {i}}}{l _ {i} ^ {\prime}} + \frac {e ^ {x _ {i} - m _ {i}}}{l _ {i} ^ {\prime}} V [ i,: ] \tag {44} \\ \end{array}
$$

Calculating Attention requires only one round of iterationswith the above formula, so we can chunk the calculation tofind the final result. FlashAttention-2 [261] improves on thisby improving the formula 44 as follows:

$$
o _ {i} ^ {\prime} = o _ {i - 1} ^ {\prime} l _ {i - 1} ^ {\prime} e ^ {m _ {i - 1} - m _ {i}} + e ^ {x _ {i} - m _ {i}} V [ i,: ] \tag {45}
$$

Compared to the original $o _ { i } ^ { \prime } ,$ we only need to divide $l _ { N } ^ { \prime }$by one more $l _ { N } ^ { \prime }$ in the final computation $o _ { N } ^ { \prime }$ to get thecorrect result, thus avoiding the intermediate multistepscaling division operation. It also reduces the memory writeoverhead. Specifically, in FlashAttention, it is fixed $K _ { j } , V _ { j }$enumeration of $Q _ { i } , { \dot { O } } _ { i } , l _ { i } ^ { \prime } , m _ { i }$ for computation; in this way,for each computed $O _ { i }$ we need to write it back to memory,which requires $O ( N ^ { 2 } d ^ { 2 } M ^ { - 1 } )$ write complexity, where $M$denotes the size of the shared memory. In FlashAttention-2, we fixed $Q _ { i } , O _ { i } , l _ { i } ^ { \prime } , m _ { i }$ to enumerate $K _ { j } , V _ { j } ,$ so that thefinal result of $O _ { i }$ can be computed at once and then writtenback to memory, and the complexity of writing is reduced to$O ( N d )$ . In addition, it also parallelizes the dimension of se-quence length; when the batch size and the number of headsare small, it increases the parallelism on the sequence lengthto improve the GPU occupancy, significantly improving thecomputation speed.

In general, efficient attention optimization methodsmainly include sparse attention, linearized attention, andFlashAttention. However, there is often a gap between thepractical and theoretical effects of many efficient attention

methods, for example, many sparse attention methods aredifficult to achieve the theoretical effects in practice dueto the discontinuous memory accesses, which is mostlybecause we do not take into account the characteristics ofthe existing hardware when improving the methods.

# 6.2 Neural Architecture Search.

Although there have been significant advances in compres-sion and acceleration methods for LLMs, many current hy-perparameters that determine the final shape of the modelstill need to be determined by hand design. This handdesign approach often requires a great deal of specializedknowledge and experience on the part of the designer, andit also has the problems of requiring long training time andhigh cost. In this dilemma, one promising solution is NeuralArchitecture Search (NAS) [262], [263], [264], [265], [266],[267], [268], [269], [270]. For simplicity’s sake, next, we willpresent a representative work from one of them.

The high computational cost of the Transformer modelmakes it difficult to deploy on some hardware devices andto realize the low-latency of inference on hardware deviceswith limited resources, HAT [262] has emerged. The ideaof HAT is to search for the best-performing model structureparameter that satisfies the requirement (given the hardwareconditions and resources) for a given latency requirement.However, searching out the model structure and trainingand evaluating it from scratch is costly and slow. It avoidsexpensive retraining by constructing a Super Transformersuch that it approximately contains all Sub Transformermodels in the search space by sharing weights. Meanwhile,HAT trains a delay predictor to predict the delay throughan offline method, which further speeds up the search. Inaddition, it observes several important properties:

First, focusing on multiple encoding layers is benefi-cial for decoding layers.

Second, different hardware has different preferencesfor the model, with GPUs preferring shallow andwide Transformers and ARM CPUs preferring nar-row and deep Transformers.

Overall, HAT provides an efficient NAS scheme for trans-former models under different hardware conditions andlatency requirements. At the same time, it can be well com-bined with other compression acceleration methods becauseit finds suitable model structure parameters for a givencondition without changing the model’s architecture.

# 7 DYNAMIC NETWORKS

Scaling up the size of language models has been proven tobe an effective approach for enhancing their performanceon NLP tasks [271], [272]. However, the substantial compu-tation costs and memory demands associated with scalingpresent a major challenge in the advancement of LLMs.To address these issues while still harnessing the benefitsof scaling, dynamic neural networks (DyNNs) engage only asubset of the network for processing each input, making theentire model more flexible and efficient in meeting computa-tional demands under resource-constrained environments.In the field of NLP and the domain of LLMs, currentresearch on DyNNs primarily encompassess the following

three methodologies: early exit, cascade inference and mixtureof experts (MoE).

Early exit is designed to dynamically terminate theinference process at the early layers of deep neural networks(DNNs), thereby reducing computational costs and improv-ing response time [273]. The intuition is that the predictionsfor less complex words can often be accurately accom-plished in earlier layers of the network [274]. These methodstypically integrate a series of internal classifiers within thenetwork, which provide signals for early exiting duringinference. Various exit criterions have been proposed [275],[276], [277], [278], [279], [280], [281], [282]. This line of workmainly focuses on and is applied to small or medium-sizelanguage models, such as Bert. And the accuracy may notbe sufficient enough to support the application of generalLLMs in more complex and realistic scenarios.

Casacade inference utilizes a series of language modelsof varying sizes to process requests with different levels ofcomplexities. Tabi [283] proposes an inference system withmulti-level inference models and a probability-based dis-patcher to determine the handling strategy for input queriesand balance both accuracy and efficiency. FrugalGPT [284]learns to adaptively triage quries from diverse datasets andtasks and direct them to an appropriate combination of LLMAPIs. Both EcoAssistant [285] and [286] employ a querycache to reference historical data for faster responses and ahierarchy of LLMs to handle those mismatched new queries.Mixture-of-Thoughts [287] considers the consistency of an-swers from weaker LLMs as an indicator of the questiondifficulty to decide whether to leverage stronger LLMs.Generally, this line of works has emerged recently anddemonstrates a promising direction for the development ofmore efficient LLM systems.

Compared to the two types of methods above, the studyof MoE has an extensive history spanning multiple machinelearning fields including NLP. MoE horizontally extends afeed-forward network (FFN) with multiple sub-networks,of which only one or few will be activated during a singleforward pass. It is widely incorporated into the architecturesof today’s LLMs [288], [289] to provide both efficient andpowerful services. So in the remainder of this section, wewill delve into the realm of MoE. Section 7.1 begins with anintroduction to the basic concepts of MoE, followed by anextensive survey of contemporary research on incorporatingMoE into LLMs, which includes algorithmic and architec-tural design, training strategies and pratical applications.Section 7.2 offers a concise review of some representativestudies on integration of MoE with previously dicussedmodel compression and acceleration techniques, highlight-ing its potential in the development of more comprehensiveand cost-efficient LLM systems.

# 7.1 Mixture of Experts

The earliest concept of MoE dates back to three decadesago [298], [299] but firstly demonstrates its effectivenessin massively improving model capacity without payinga proportional computation overhead using sparse gating[292]. In sparse MoE models, a subset of model parametersare partitioned into a set of $N$ expert networks $\mathsf { \bar { \{ E } }  _ { i } ( \cdot ) \} _ { i = 1 } ^ { N } ,$each operates independently on the input with unshared


TABLE 5: A summary of various Mixture-of-Experts (MoE) methods. For models of the largest size, we present the totalnumber of paremeters along with the number of experts per MoE layer. For methods utilizing shared experts [290], [291],we include both (the number of experts used for sharing $^ +$ the number of experts used for routing).


<table><tr><td>Methods</td><td>Base Model</td><td>Sparsity</td><td>Largest Model Size (Params / Num. experts)</td><td>Load Balance</td></tr><tr><td>Sparsely-Gated [292]</td><td>LSTM</td><td>top-k</td><td>137B / 131072</td><td>Noisy top-k gating and auxiliary loss term.</td></tr><tr><td>GShard [293]</td><td>NMT</td><td>top-2</td><td>600B / 2048</td><td>Local group dispatching and auxiliary loss term.</td></tr><tr><td>Switch [294]</td><td>T5</td><td>top-1</td><td>1571B / 2048</td><td>Auxiliary loss term.</td></tr><tr><td>Expert Choice [295]</td><td>Transformer</td><td>Expert Choice</td><td>143B / 64</td><td>Expert choice routing.</td></tr><tr><td>DeepSpeed-MoE [290]</td><td>GPT-3</td><td>Residual-MoE</td><td>52B / (1+127)</td><td>Multi-expert and multi-data parallelism.</td></tr><tr><td>M6-T [296]</td><td>M6</td><td>k top-1</td><td>10003B / 960</td><td>-</td></tr><tr><td>Brainformer [297]</td><td>Non-uniform</td><td>Expert Choice</td><td>158B / 64</td><td>Expert choice routing.</td></tr><tr><td>Mixtral 8x7B [289]</td><td>Mistral 7B</td><td>top-2</td><td>47B / 8</td><td>-</td></tr><tr><td>DeepSeekMoE [291]</td><td>DeepSeek</td><td>Shared Experts</td><td>145B / (4+128)</td><td>Auxiliary loss terms.</td></tr></table>

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-11/0a775777-6cdd-4dbe-a304-295f64b9a1ad/5a96063b5298fc702b75cb8558e8e31ab0d8883c425131425fae4def01c86f91.jpg)



Fig. 8: Illustration of a transformer block with an integratedMoE layer.


weight. During training and inference, each input example$x$ (i.e., a token representation in language models) would berouted to specific expert(s) via gating function $G ( \cdot )$ whoseinput is also $x$ and the output is a sparse $n$ -dimensionalvector. The final output $y$ of MoE module is a weightedcombination which can be written as:

$$
y = \sum_ {i = 1} ^ {N} G (x) _ {i} E _ {i} (x) \tag {46}
$$

Based on the sparsity of $G ( x )$ , we can skip the computationof $E _ { i } ( x )$ wherever $G ( x ) _ { i } = 0$ . Since only part of modelparameters are activated for each input and all experts havethe potential to be utilized by different samples, MoE modeltheoretically enjoys a competitive learning ability with amuch faster inference speed comparing to its dense coun-terpart. As Transformer [1] has become the standard choicefor language model, the most common way to introduceMoE into Transformer is to replace the feed-forward layerof certain block(s) with MoE layer, of which each expertis itself a regular Transformer feed-forward network. Anexample of MoE layer is given in Fig. 8. By increasing thenumber of experts, the parameter size could grow fromhundreds of millions to hundreds of billions or even trillionsto match the size of LLMs. There are mainly three key

elements to characterize one MoE method: (1) Routingmethod decides where and how each input will be routedin the MoE layer, which is the most critical element of MoEalgorithms. (2) MoE model architecture discusses commonor specific design choices towards building scaled modelsthat are more performance-effective and parameter-efficient.(3) Special training strategies sometimes are needed toaccommodate the uncertainty raised from learning-basedrouting methods. We summarized some representative MoEmethods in TABLE 5.

# 7.1.1 Routing Method

For an MoE system, the most crucial factor affecting its per-formance and the primary design concern is the ensuranceof load balancing among the experts.

In a standard distributed training settings, experts fromthe same layer are scattered across multiple devices. Each ofthem is a simple feed-forward network (FFN) in most casesand computes in parallel. And the maximum number oftokens each expert can process during a single forward pass,also known as the expert capacity, is limited by the memoryof device it resides. Generally, without specific algorithm orarchitecture design, tokens from the same batch are assignedunevenly among experts by the gating function due to itssparsity. Therefore, if too many tokens are routed to thesame expert, surpassing its capacity, this will lead to anoverflow issue. The computation for the overflown part willbe skipped and those tokens will be passed directly to thenext layer via a residual connection. Thus unbalance loadsacross experts lead to under-processing of tokens and awaste of computation and memory for experts with emptyslots.

Besides performance decline, imbalanced loads acrossexperts could also lead to a self-reinforcing phenomenon thatinherently limits the capabilities of MoE system throughtraining. This phenomenon manifests as the gating net-work converging to a state where it always produces largeweights for the same few experts, which rapidly trainedwith large amount of data and are favored by the gatingnetwork even more. Consequently, the remaining expertsremain undertrained and underutilized, which results inthe original MoE network collapses to a smaller networkcomprising only the few active experts.

Alongside the problem of load imbalance, there arealso methods dedicated to mitigate other adverse effectsstemming from the sparse nature of MoE, such as unstablerouting or the trade-off between sparsity and accuracy.

Based on the primary problem each method address, wecategorize the existing routing methods into the followingtwo groups and begin our review.

Towards load balancing. Most routing methods applyan additional learnable gating network in each MoE layer.A simple choice yet adopted by most works is to use onelinear layer with trainable matrix $W \in \mathbb { R } ^ { d \times N } ,$ , where $d$ ismodel dimension and $N$ is the number of experts, thenfollowed by a non-linear function like softmax or sigmoid.The $N$ columns of $\textit { W } \left\{ w _ { 1 } , \cdots , w _ { N } \right\}$ can also be seen asembeddings of $N$ experts respectively and readers may findthis expression in some works. For each token $x \in \mathbb { R } ^ { d }$ , therouting score between $x$ and $i$ -th expert is given by the dot-product similarity metric $s _ { i } = x \cdot w _ { i }$ . To add sparsity to thegating network, i.e., to use onlylayer, only experts with the hi $k$ expertsest top- $k$ $( k \ll N )$ ea of $\{ s _ { i } \} _ { i = 1 } ^ { N }$(set of indices $\tau$ ) will be selected for token $x$ . In general, ifexpert $i$ is activated, its gating value is given by

$$
G (x) _ {i} = \left\{ \begin{array}{l l} \exp \left(s _ {i}\right) / \sum_ {j \in \mathcal {T}} \exp \left(s _ {j}\right), & \text {s o f t m a x g a t i n g}, k > 1 \\ \exp \left(s _ {i}\right) / \sum_ {j = 1} ^ {N} \exp \left(s _ {j}\right), & \text {s o f t m a x g a t i n g}, k = 1 \\ \sigma \left(s _ {i}\right), & \text {s i g m o i d g a t i n g} \end{array} \right. \tag {47}
$$

where $\sigma ( \cdot )$ is the sigmoid function. Note that for softmaxgating, top-1 methods formulate sightly differently to make$G ( x ) _ { i }$ non-trivial. The above idea is first proposed in [292]and applied to LSTM models [300]. To mitigate the issueof self-reinforcing, they add a tunable Guassian noise tothe routing scores and employ two additional loss termsto encourage more balanced routing scores and selectionrates across the experts. GShard [293] integrates MoE intotransformers by replacing every other feed-forward layerwith an MoE layer using top-2 gating. They propose severalstrategies to ensure load balancing, including (1) partion-ing tokens into groups and limiting the number of tokensthat each experts can receive from a single group; (2) anauxiliary loss term which has been widely adopted by laterworks [294], [301], [302] and (3) a random routing strategy.To further align the computational cost with that of a vanillatransformer, Switch Transformer [294] routes each token toonly one expert with top-1 gating.

The aforementioned studies primarily suggest interme-diary strategies, such as auxiliary loss terms, to promoteload balancing during training. Other researches aim atimproving the gating function to directly regularize oreven guarantee perfectly balanced loads across experts.BASE Layer [303] formulates token-to-expert allocation asa linear assignment problem, which maximizes the scoresbetween expert and their assigned tokens while subject toa constraint that each expert must receive an equal amountof tokens. Expert choice [295] is another routing strategyguaranteeing perfect load balancing with no additionalregularization required. Instead of letting tokens select thetop- $k$ experts, each expert chooses top- $k$ tokens which alsobased on routing scores. Consequently, each expert can haveexactly same workloads and each token can be processedby a variable number of experts. Clark et al. [304] propose a

routing algorithm using reinforcement learning, in whicheach router is seen as policy with actions and rewardsdefined by the selection of experts and the predicted proba-bility of the correct output token respectively. In addition toemploying learnable gating networks, some studies imple-ment non-learnable strategies to regulate the load distribu-tion across experts. Hash layer [305] employs a parameter-free hash function to replace dynamic gating with a pre-defined fixed mapping from tokens to specific experts,which consequently eliminates load imbalance. MoWE [306]routes each word to one specific expert based on auxiliaryvocabulary mapping and ensures that the words assignedto each expert are of approximately the same frequencyin the pretraining data. Inspired by Hash layer, PanGu-$\Sigma$ [307], [308] deploys a two-level routing that first mapseach token to a group of candidate experts by domain andthen uses random hash to choose a particular expert fromthat group for processing. THOR [309] also simplifies thegating process with a parameter-free approach by randomlyselecting a pair of experts from each layer during a trainingiteration and dispatching the whole batch of tokens to thoseexperts.

The methods we have described so far can only beutilized to the fullest extent with sufficient device resources(i.e., no overflow issues are encountered). In more realisticsenarios, one must take additional strategy to handle over-flow. The simplest solution is to stop assigning tokens toexperts with full workloads. However, in this way onlyeither the prefix of the input sentences or the sentenceswith small indices in the batch dimension will be processed,based on whether the batch is flattened along the sequencelength or the batch dimension, which leads to a biasedselection and underutilization of training or inference data.To address this issue, Z-code [310] and BASE [303] pro-pose to shuffle the input tokens in the flattened batch todisentangle the probability of a token being selected fromits position. In the domain of vision transformer, V-MoE[311] introduces Batch Prioritized Routing algorithm (BPR),which additionally compute a priority score for each token(e.g., the maximum of routing scores between the token andall experts) and sort tokens accordingly before allocation.Therefore only insignificant tokens will be discarded. ST-MoE [312] finds that BPR benefits language models aswell, especially in low-resource regimes where the expertcapacity is even less than the average number of tokenseach expert receives. Note that BPR can only be applied toencoder side of encoder-decoder model since the inputs ofencoder are not autoregressive thus are allowed to see eachother, otherwise model could cheat by using future tokeninformation.

Effective routing with sparsity. Although sparsitybounds the computational cost in a large-scale MoE system,it generally limits the network’s capability and impedesconvergence due to unstable routing dynamics. DeepSpeed-MoE [290] observes that increasing the number of expertseach token goes through helps accuracy. In order to leveragethis property while keeping the computation costs as top-1 gating, they propose Residual-MoE by fixing one expertand varying the second expert for each token to achieve thebenefit of using 2 expert. DeepSeek-MoE [291] employs thesame methodology, utilizing a fixed set of shared experts

to capture and consolidate common knowledge, augmentedby a distinct set of routed experts dedicated to knowledgespecialization. M6-T [296] also notices the advantage of top-$k$ gating over the top-1. They propose to split expert into $k$groups and perform $k$ top-1 routing in parallel to matchthe efficiency of top-1 gating. To ensure each expert canreceive rich and diverse tokens under the sparse settings,MoEC [302] encourages experts to form clustered structureby closing the routing probability among neighbor expertswith designed loss term and randomly drops some expertsin each cluster before the global top-1 gating.

Besides sparsity, DSelect-k [313] notices the discontinuityin top-k gating methods and suggest this could lead toconvergence and performance issues when training withgradient-based methods. They propose a continuously dif-ferentiable and sparse gating function, which densely se-lects experts at beginning of training but fast converges tosparse expert selection by adding an regularization term. X-MoE [301] points out that current routing mechanisms tendto push token representations towards expert embeddingswhich potentially harms the representation capacity. Theypropose to calculate the routing scores between tokens andexperts in a low-dimensional space with additional nor-malization and learnable gating temperature. ST-MoE [312]conducted an extensive study on the training stability andfine-tuning quality of sparse MoE models. They framedthe training process of sparse models as a quality-stabilitytrade-offs: various stability techniques, such as dropout andgradient clipping, could enhance the training stability butoften come at the expense of model quality. Therefore theypropose router z-loss to address both the instability issueand the problem of quality degradation.

# 7.1.2 MoE model architecture

In this part we discuss how to arrange MoE layers into aTransformer-based model, such as the frequency of expertlayers and the number of experts, which could significantlyaffect the scale of models and the overall performances.

In transformer, sparse model usually starts with a densemodel and scales up by substituting or inserting MoE lay-ers at a fixed interval or heuristically. A common designdeployed in most large sparse expert models [293], [294],[295], [314], [315] is to replace the feed-forward componentof every other Transformer block with a MoE layer (i.e., at afrequency of 0.5). Other frequencies are also adopted, suchas 0.25 (i.e., substituting every fourth FFN layer) in [312] and1.0 (i.e., placing in every layer). In general, experiments [304]suggest a frequency at 0.5-1.0 and lower frequency under-mines the performance. However, there are also works [301],[302], [303], [305], [316] introducing a fixed number of expertlayers to baseline models by spreading MoE layers unevenlyacross the network. For instance, BASE [303] inserts a largeMoE layer consisting of stacked FFNs only after middlelayer.

As for the number of experts per-layer, although usingmore experts continuously brings quality gains in mostcases, diminishing returns in the improvements are alsoreported in earlier works [293], [294]. Further analyses [304],[312] also points out the drastically-diminishing incrementalbenefits from routing as the base model size increases. A

fixed number of experts per-layer in $\{ 6 4 , 1 2 8 \}$ is recom-mended by [304] and also have been practiced in a lot ofsparse large language models [312], [314], [317]. Moreover,DeepSpeed-MoE [290] questions the standard MoE architec-ture putting the same number of experts in all MoE layers.Their experiments suggest that a large number of experts indeeper layers boost the performance more effectively. There-fore they introduce a pyramid structure by utilizing moreexperts only in the last two layers and achieve comparableresults as standard MoE models but with fewer parameters.

Above works are all built upon uniform transformerblocks and by interleaving dense and sparse layers, Brain-former [297] explores a non-uniform architecture withsparse layer inspired by the success of EfficientNet [318]and sandwich transformer [319]. An evolutionary searchalgorithm is applied to explore the best Brainformer blockarchitecture in the search space consisting of different layertypes (namely self attention, MoE and dense FFN sub-layers), interleaving orders and hyperparameter settingssuch as model dimension and number of attention heads.The whole network is constructed by stacking a variablenumber of blocks according to different scales. Experimentresults demonstrate a clear advantage in terms of bothefficiency and capacity over its GLaM [314] counterpart andPrimer [320] dense model produced by NAS [321].

# 7.1.3 Training Strategies

Existing learning-based routing methods usually train boththe gating and expert networks jointly from scratch. Asthe parameters of MoE layers are randomly initialized, therouting behavior at the beginning stage of training can beseen as random routing and the correspondences betweentokens and experts are highly unstable. As a result, MoEmodels take a longer time to converge with a potential riskof reinforcing improper routing behavior which eventuallylimits the model quality.

To handle this problem, a two-stage training strategy isintroduced in [316], [322] to separate the training of thegating network and expert networks. In the first stage,StableMoE [316] learns a balanced and cohesive routingstrategy following a standard MoE training process with ad-ditional balance loss. Throughout the second stage, the rout-ing strategy is freezed to provide a stable token-to-expertassignment for the training of the rest of model. Experimentsconfirm that the consistency of routing strategy boost boththe convergence speed and final performance. Conversely,EvoMoE [322] starts from training then diversifying fromone common expert at stage one before learning the gatingnetwork and sparsifying the network in the second stage.In this way experts are able to get sufficient training in theearly stage and more suitable routing strategy can be builton the exploitation of specialized experts.

Another line of works set out to alleviate the overfittingproblem raised from the imbalance between vast numberof parameters and limited training examples via specialdropout [323] mechanism at MoE layer. For instance, SwitchTransformer [294] increases the dropout rate solely insidethe experts, named as expert dropout, to aid the performanceon downstream tasks with very few training data. GatingDropout [324] further pushes traditional dropout to anotherlevel to reduce the communication cost for dispatching

tokens across devices and also improve the performanceduring training. Specifically, they permit tokens to ignorethe assignment from gating network with certain probabilityand instead route them to the experts on the same device.This also encourages experts to function more robustly andlearn a generalization ability. The results demonstrate thatGating Dropout indeed accelerate the convergence of sparseMoE models in terms of wallclock time and enhance themodel quality.

# 7.1.4 MoE Applications

The success of MoE models promotes a series of worksdeploying sparse MoE algorithms in actual LLM applica-tions or combining with other model compression and ac-celeration techniques in pursuit of greater efficiency. CPM-2[325] integrates BASE Layer [303] into their largest Chinese-English bilingual models with 198 billion parameters. Fol-lowing GShard [293], GLaM [314] trains a family of decoder-only sparse MoE models, the largest of which has 1.2T pa-rameters and yields better zero, one and few-shot learningabilities in comparison with its dense GPT-3 [9] counter-parts. Seeking to close the performance gap between highand low-resource languages and break the 200 languagebarrier, a Sparsely-Gated MoE model with 54.5B parametersis developed by [317], following the optimaztion processin [293], and casts light on a promising approach towardsa universal translation system. A revealing article on thetechnical details of GPT-4 [288] confirms the deploymentof a MoE model consisting of 16 experts inside GPT-4.Each expert is tuned to specialize in a specific domain ortask, thereby endowing GPT-4 with the multi-task ability.Mixtral [289] builds upon Mistral 7B and replaces all FFNsub-blocks by MoE layers, each consisting of 8 experts anda top-2 gating network. The resulted Mixtral 8x7B onlyuses 13B active parameters during inference but surpassesLlama 2 70B [18] and GPT-3.5 on several benchmarks.DeepSeekMoE [291] proposes a series of MoE models withsizes of 2B, 16B, and 145B as well as aligned versions, todemonstrate the adaptability and versatility of their MoEarchitectures. OpenMoE [326] also releases a suite of open-sourced MoE models, building upon the architectures of ST-MoE [312] and Residual-MoE [290]. In addition, they offer acomprehensive study and a few insights on MoE’s routingbehavior throughout training.

# 7.2 Combineing MoE with other efficient techniques

The Mixture-of-Experts approach inspires the field as analternative pathway for buidling more powerful and effi-cient LLMs. Given that MoE is akin to an art of architecturedesign and is orthogonal to most model compression andacceleration techniques, there are also works exploring waysto merge its inherent sparsity with other optimization strate-gies, such as pruning, distillation, and PEFT. In this section,we will examine the most representative studies in this areaand highlight the potential it holds for future research.

# 7.2.1 Model Compression

In the realm of sparse MoE models, most existing works canbe viewed as trading memory consumption for model qual-ity. To reduce the memory footprint while retaining most

of their capabilities, researchers have explored several waysto introduce traditional model compression techniques intoMoE models.

Switch Transformer [294] made the first attempt to distilllarge sparse models into small dense models. Their findingsindicate that it is possible to preserve approximately $3 0 \%$ ofthe quality gains achieved through scaling when distillingto a FLOP-matched dense variant for both pre-training andfine-tuning tasks. DeepSpeed-MoE [290] studies the poten-tial of distilling a large teacher MoE model into a smallerstudent MoE model with shallower expert networks. Ad-ditionally, they suggest a stage-KD training strategy (i.e.,halting distillation at specific steps) to mitigate the under-fitting issue stemming from the student model’s limitedexpert capacity.

As another prominent tool for parameter reduction, thepruning of MoE models aims to remove redundant or lessinfluential components, usually a subset of less importantexpert networks, with minimal impact on the performance.The hypothesis behind pruning, as suggested by Z-code[310], is that different experts can specialize in distinct as-pects of the task during training, making a subset of expertscompetent enough for a given task to a certain extent. Z-code tried two methods for expert selection: random selec-tion and selection based on utilization rates in the validationset. Chen etc [327] observe the long-tailed distribution ofexpert contributions in downstream tasks. Different from Z-code’s approach, they propose to progressively prune awaymost experts throughout the fine-tuning process, leavingonly the most professional one for the target downstreamtask. The experiment results highlight the effectiveness oftheir pruning strategy, preserving $9 9 . 3 \%$ of the benefits fromMoE while enjoying the same resource consumption asvanilla dense models during inference. As an alternativeapproach, MPOE [328] introduce a parameter-efficient MoEarchitecture by decomposing the parameter matrix of eachexpert into central and auxiliary tensors. The central tensor isbelieved to encode the majority of the information presentin the original matrix, which is likely to be similar acrossexperts and thus suitable for sharing among them. On theother hand, the auxiliary tensors capture the individualcharacteristics and serve as a complement to the centraltensor. This parameter-sharing method has been shown tobe effective, achieving a $2 7 . 2 \times$ reduction in total parameterswhile yielding better performance comparing to the SwitchTransformer.

Witnessing the great success of MoE models, there arealso efforts to introduce sparsity into a standard transformermodel with the purpose of reducing the number of param-eters involved in computation while retaining the repre-sentation power. MoEBERT [329] adapts the feed-forwardnetworks in a pre-trained BERT [3] into multiple expertsand activates only one expert during inference to increasethe speed. To preserve the original representation power,they share the most important neurons in the FFNs amongthe experts based on the importance score [330] when ini-tializing MoEBERT. The training of MoEBERT incorporateslayer-wise distillation, leading to a resulting model that out-performs other task-specific distilling methods. MoEfication[331] aims to generalize the conversion from FFNs to MoElayers for various Transformer models. The idea is driven

by the insight that only a tiny fraction of neurons of FFNswill respond to most inputs. To split the feed-forward layerinto experts, neurons that often activates simultaneously aregrouped into the same expert network. And the routingstrategy is learned by approximating the calculation of theoriginal model. To further reduce the computational andmemory demards of standard transformers, $\sigma$ -MoE [332]and SwitchHead [333] introduce additional sparsity to theFFN and attention components, drawing on the principlesof the MoE methodology.

# 7.2.2 Efficient Finetuning

In search of more efficient and powerful model architec-tures, researchers are also exploring the combination of MoEmethods and other cost-effective techniques such as Mixer[334] and PEFT methods. These collaborative approachesprimarily leverage the expressiveness provided by MoEwhile aggressively reducing the training and computationcost. Sparse Mixers [335] replaces most of the self-attentionsublayers with mixing and FFNs with MoE sublayers. SMLP[336] goes one step further by substituting the self-attentionmodules with linear transformations, which also employsa MoE mechanism with routing in the feature dimensionto ensure tokens from the same sentences are deliveredto the same expert. AdaMix [337] proposes a mixture ofadapters [338] or a mixture of low-rank decompositionmatrices [339] with stochastic routing mechanism [309] asa novel fune-tuning technique to enhance the downstreamperformance. The result illustrates that AdaMix surpassesSOTA parameter-efficient fine-tuning and even full modelfine-tuning algorithms on both NLU and NLG tasks. Basedon a similar idea, MixDA [340] also utilizes a set of domainadapters to inject domain-specific knowledge in parallel andthen train a mixture-of-adapters gate to dynamically fusethe knowledge from different domain adapters. This plug-in approach showcases its scalibility and efficiency on sev-eral domain tasks. The same methodology is also adoptedby [341], [342], [343], [344] to achieve efficient finetuning ondomain-specific or instruction datasets and to mitigate thecatastrophic forgetting arising from continual learning.

# 8 ACCELERATION FRAMEWORK

With the rapid development of Transformer-based models,various models have emerged. Because of different applica-tion scenarios, they have additional requirements in termsof latency, throughput, memory, etc, making it difficult forus to deploy the models. In this section, we introduce somerecently developed inference acceleration frameworks forLLM, which effectively improve the efficiency of the modelsin different scenarios, as shown in TABLE 6. We classifythe general framework and specialized framework based onthe generalization. Here are some more acceleration frame-works [351], [352], [353], [354], [355], [356], [357] specific totraining, and since this paper focuses on inference, we willnot discuss them specifically. If you want to deploy trainedmodels to get efficient inference quickly, you can refer tothese frameworks [358], [359], [360], [361], [362], [363].

# 8.1 General Framework

In this section, we will introduce some relatively general-ized frameworks [345], [346] proposed recently. No matterwhat kind of scenarios the models are deployed in, we canconsider using them or combining their ideas to acceleratethe inference and thus obtain higher efficiency. Since mostbig models are still deployed and run on GPUs, our gener-alization here refers to generalization under GPU hardware.

Operator fusion is a common method to acceleratemodel inference by eliminating unnecessary intermediateresults, lowering memory requirements, and reducing un-necessary memory IO and kernel startup overhead, thusimproving the utilization of computational resources suchas GPUs, CPUs, and registers. Meanwhile, operator fusionis an essential optimization for many state-of-the-art DNNcompilation frameworks, such as TensorFlow XLA [364],TVM [365], MNN [366], PyTorch JIT [367], and so on.However, these frameworks have stringent requirementsfor operator fusion, e.g., TVM uses relatively fixed sched-ule templates, resulting in missing many potential fusionopportunities. DNNFusion [345] has better coverage andfusion identification capabilities through algebraic simpli-fication and rational classification of operators. In addition,it further improves the efficiency of operator fusion by elim-inating unnecessary operators through heuristic methods.

Recently, Microsoft proposed DeepSpeed Inference [346],an efficient integrated inference system for the increasinglydiverse Transformer model, which reduces latency by afactor of 7.3 in state-of-the-art latency-oriented scenariosand increases throughput by more than 1.5 in throughput-oriented scenarios. It includes the following two compo-nents:

Multi-GPU Inference Solution: It minimizes latencywhile maximizing throughput in dense and sparseTransformer models with GPU memory aggregation.

Heterogeneous Inference Solution: Besides GPUmemory and computation, it utilizes CPU and NVMememory to achieve high inference throughput forlarge models that do not lend themselves to aggre-gated GPU memory.

Many strategies have been used to maximize the trainingthroughput, such as tensor parallelism, pipeline parallelism,ZeRo, expert parallelism, and so on. However, inferencewith small batch size suffers from the following problemsdue to insufficient parallelism:

A smaller amount of data is processed at a time,which results in the need to read model weights fromthe HBM and call the kernel frequently, incurring asignificant overhead.

Each kernel call writes data to global memory, whichthe GPU has to re-read on the next kernel call, addingadditional communication overhead.

The current cuBLAS and CUTLASS GeMM librariesare not optimized for small batch sizes and have lowmemory bandwidth utilization.

On the other hand, regular operator fusion can onlybe done for element-wise operators. In contrast, operatorsin the Transformer structure introduce data dependenciesacross thread blocks, making it challenging to do operator


TABLE 6: A summary of various acceleration frameworks.


<table><tr><td>Framework/Passage</td><td>Generalization</td><td>Method</td></tr><tr><td>DNNFusion [345]</td><td>General</td><td>Operator fusion</td></tr><tr><td>DeepSpeed Inference [346]</td><td>General</td><td>Operator fusion, tensor parallelism, inference pipeline</td></tr><tr><td>TurboTransformer [347]</td><td>Specialized</td><td>Operator fusion, scheduling optimization</td></tr><tr><td>ByteTransformer [348]</td><td>Specialized</td><td>Operator fusion, scheduling optimization</td></tr><tr><td>FlexGen [349]</td><td>Specialized</td><td>Offloading system</td></tr><tr><td>Power-Infer [350]</td><td>Specialized</td><td>Offloading system</td></tr></table>

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-11/0a775777-6cdd-4dbe-a304-295f64b9a1ad/5e7b9d9c1b07ae3a6eedcd392974add8089448ec676b45fe7af9c38e9303831f.jpg)



Fig. 9: Deep-Fusion strategy for the small-batch inference.


fusion. This is because if another consumes data generatedby one thread block on GPUs, a global memory synchroniza-tion is needed to invoke a new kernel. To avoid the need forglobal synchronization, Deep-Fusion tiles the computationalspace along the dimensions of the iteration space so that nocross-block data dependencies arise. In addition, it custom-designed GeMM for small batch sizes and implementedit to be fusible with Deep-Fusion for maximum memorybandwidth utilization. Deep-Fusion does four operator fu-sions in a Transformer layer for small batch sizes to obtainfour customized kernels, as shown in the red dashed box inthe Fig. 9 below. For large batch sizes, the operator fusionstrategy is the same, with the difference that the GeMM incuBLAS is used directly instead of the custom GeMM.

# 8.2 Specialized Framework

In this section, we will introduce some specialized frame-works that have been proposed recently. They are cus-tomized for specific scenarios and needs and can be tailoredto meet different requirements. If you deploy models indi-vidually in high demand in certain aspects, consider usingthem.

Compared to other scenarios, efficiently deploying theTransformer model to servers needs to meet the service’slow latency and high throughput requirements, whichpresents a significant challenge. In addition, due to theunpredictability of requests and the fact that NLP tasksemploy variable-length sentences, the variability of input di-mensions poses a severe problem for effective memory man-agement and service optimization. TurboTransformer [347]proposes a sequence-length-aware memory allocation algo-rithm and a batch scheduling algorithm that aims to max-imize the response throughput by treating it as a dynamicprogramming problem. Efficient memory reuse of variable

dimensional intermediate tensor and optimal batch schedul-ing scheme are realized. TurboTransformer also proposes aparallel approximation algorithm for high-frequency opera-tors such as Softmax and LayerNorm, significantly improv-ing efficiency. However, TurboTransformer’s active group-ing approach still introduces non-eliminable padding over-heads. Based on this, ByteTransformer [348] proposes apadding-free algorithm to free the whole Transformer fromthe redundant computation of zero-padded tokens. In ad-dition, ByteTransformer optimizes multi-head attention forthe zero-filling algorithm so that the attention is no longerfaced with redundant computation of useless tokens, furtherimproving performance.

Unlike the previous work, FlexGen [349] sacrifices the la-tency of the inference computing service almost completelyto polarize the design of an LLM computing system thatfocuses only on throughput. Thus, it is only suitable foroffline computing. Every aspect of the LLM accelerator canbe reconsidered when pursuing only throughput, includingstorage management, latency-hiding design of memory ac-cesses in cross-domain memory hierarchies, and paralleliza-tion strategies. FlexGen proposes a new offloading-basedinference system based on a zig-zag parallelization strategy,which achieves more than 40 times higher throughput thanDeepSpeed Zero-Inference. We believe the most inspiringaspect of this work is that it highlights the importance ofdivergent thinking and emphasizes the need to dig deeperinto the details of the problem and explore alternativesolutions.

Recently, the open-source inference framework Power-Infer [350] recently made LLM Inference 11 times faster.Without quantization and with FP16 precision, it allows 40Bmodels to run smoothly on an RTX4090 PC; if quantizationis added, a 2080 Ti can also run 70B models smoothly. It isbased on highly localized sparse activation based on LLM,i.e., a small fraction of neurons hot neurons are activatedall the time on input, while the majority of neurons coldneurons respond according to a specific input. PowerInferexploits this feature and the fact that CPUs are good at con-ditional computation and GPUs are good at simple parallelcomputation to develop an innovative GPU-CPU hybridinference engine. This means hot neurons are preloadedinto the GPU for fast access. In contrast, cold neurons arecomputed on the CPU, dramatically reducing the memoryrequirements of the GPU and the amount of data transferredbetween the CPU and the GPU. In addition, PowerInfer in-corporates an adaptive predictor and neuron-specific sparseoptimizations to improve the sparsity efficiency of neuronactivation and computation. Overall, PowerInfer enables PCusers to run advanced LLM locally without needing expen-

sive specialized hardware. This facilitates the popularizationof AI applications and provides unprecedented opportuni-ties for hobbyists, researchers, and small businesses.

There are already more accelerators built for large modelmulti-GPU distributed inference, but relatively few acceler-ators are built for edge device deployment. As the demandfor deploying AI large models on edge devices grows, thiswill become a pressing problem.

# 9 CONCLUSIONS

In this paper, we conducted a comprehensive investigationof compression and efficient inference for large languagemodels from an algorithmic perspective, including quan-tization, pruning, distillation, compact architecture design,dynamic networks. Additionally, we introduced some pop-ular compression and acceleration frameworks tailored forlarge language models. However, as we mentioned in theintroduction, compression and acceleration of large modelsface more challenges compared to smaller models. Whileexisting algorithms have made significant efforts to addressthese challenges, many algorithms still rely on frameworksdesigned for compressing small models, and challenges incompressing large models persist. In the future, furtherexploration is needed to develop more efficient and effectivecompression algorithms while ensuring the versatility andgeneralization of large models.

# CONTRIBUTORS

Wenxiao Wang is responsible for this paper’s overall struc-ture, content arrangement, the writing of Section 1 andSection 2, and refinement of each section in this paper.Wei Chen, Yongliu Long, Zhengkai Lin and Liye Zhangare responsible for the surveys and writing of quantization(Section 3), pruning (Section 4), dynamic networks (Sec-tion 7), and knowledge distillation (Section 5), respectively.Yicong Luo is responsible for surveys and writing compactarchitecture design and acceleration framework (Section 6and Section 8). All co-first authors are listed in alphabeticalorder of their surnames. Binbin Lin, Deng Cai, and XiaofeiHe participate in the comprehensive discussion and providemany great insights.

# REFERENCES



[1] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N.Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,”Advances in neural information processing systems, vol. 30, 2017.





[2] C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena,Y. Zhou, W. Li, and P. J. Liu, “Exploring the limits of transferlearning with a unified text-to-text transformer,” The Journal ofMachine Learning Research, vol. 21, no. 1, pp. 5485–5551, 2020.





[3] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “Bert: Pre-training of deep bidirectional transformers for language under-standing,” arXiv preprint arXiv:1810.04805, 2018.





[4] A. Radford, K. Narasimhan, T. Salimans, I. Sutskever et al.,“Improving language understanding by generative pre-training,”2018.





[5] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskeveret al., “Language models are unsupervised multitask learners,”OpenAI blog, vol. 1, no. 8, p. 9, 2019.





[6] L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. L. Wainwright,P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, J. Schulman,J. Hilton, F. Kelton, L. Miller, M. Simens, A. Askell, P. Welinder,P. F. Christiano, J. Leike, and R. Lowe, “Training language modelsto follow instructions with human feedback,” in Advances inNeural Information Processing Systems 35: Annual Conference onNeural Information Processing Systems 2022, NeurIPS 2022, NewOrleans, LA, USA, November 28 - December 9, 2022, S. Koyejo,S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, Eds.,2022.





[7] T. Lin, Y. Wang, X. Liu, and X. Qiu, “A survey of transformers,”CoRR, vol. abs/2106.04554, 2021.





[8] S. Islam, H. Elmekki, A. Elsebai, J. Bentahar, N. Drawel, G. Rjoub,and W. Pedrycz, “A comprehensive survey on applications oftransformers for deep learning tasks,” CoRR, vol. abs/2306.07303,2023.





[9] T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhari-wal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell et al.,“Language models are few-shot learners,” Advances in neuralinformation processing systems, vol. 33, pp. 1877–1901, 2020.





[10] Y. Wang, H. Chen, Y. Tang, T. Guo, K. Han, Y. Nie, X. Wang,H. Hu, Z. Bai, Y. Wang, F. Liu, Z. Liu, J. Guo, S. Zeng, Y. Zhang,Q. Xu, Q. Liu, J. Yao, C. Xu, and D. Tao, “Pangu-π: Enhancinglanguage model architectures via nonlinearity compensation,”CoRR, vol. abs/2312.17276, 2023.





[11] Z. Zhang, X. Han, H. Zhou, P. Ke, Y. Gu, D. Ye, Y. Qin, Y. Su,H. Ji, J. Guan, F. Qi, X. Wang, Y. Zheng, G. Zeng, H. Cao, S. Chen,D. Li, Z. Sun, Z. Liu, M. Huang, W. Han, J. Tang, J. Li, X. Zhu,and M. Sun, “CPM: A large-scale generative chinese pre-trainedlanguage model,” AI Open, vol. 2, pp. 93–99, 2021.





[12] T. L. Scao, A. Fan, C. Akiki, E. Pavlick, S. Ilic, D. Hesslow,R. Castagne, A. S. Luccioni, F. Yvon, M. Gall ´ e, J. Tow, A. M. Rush, ´S. Biderman, A. Webson, P. S. Ammanamanchi, T. Wang, B. Sagot,N. Muennighoff, A. V. del Moral, O. Ruwase, R. Bawden, S. Bek-man, A. McMillan-Major, I. Beltagy, H. Nguyen, L. Saulnier,S. Tan, P. O. Suarez, V. Sanh, H. Laurenc¸on, Y. Jernite, J. Launay,M. Mitchell, C. Raffel, A. Gokaslan, A. Simhi, A. Soroa, A. F. Aji,A. Alfassy, A. Rogers, A. K. Nitzav, C. Xu, C. Mou, C. Emezue,C. Klamm, C. Leong, D. van Strien, D. I. Adelani, and et al.,“BLOOM: A 176b-parameter open-access multilingual languagemodel,” CoRR, vol. abs/2211.05100, 2022.





[13] S. Zhang, S. Roller, N. Goyal, M. Artetxe, M. Chen, S. Chen,C. Dewan, M. T. Diab, X. Li, X. V. Lin, T. Mihaylov, M. Ott,S. Shleifer, K. Shuster, D. Simig, P. S. Koura, A. Sridhar, T. Wang,and L. Zettlemoyer, “OPT: open pre-trained transformer lan-guage models,” CoRR, vol. abs/2205.01068, 2022.





[14] A. Zeng, X. Liu, Z. Du, Z. Wang, H. Lai, M. Ding, Z. Yang, Y. Xu,W. Zheng, X. Xia et al., “Glm-130b: An open bilingual pre-trainedmodel,” arXiv preprint arXiv:2210.02414, 2022.





[15] A. Chowdhery, S. Narang, J. Devlin, M. Bosma, G. Mishra,A. Roberts, P. Barham, H. W. Chung, C. Sutton, S. Gehrmann,P. Schuh, K. Shi, S. Tsvyashchenko, J. Maynez, A. Rao, P. Barnes,Y. Tay, N. Shazeer, V. Prabhakaran, E. Reif, N. Du, B. Hutchinson,R. Pope, J. Bradbury, J. Austin, M. Isard, G. Gur-Ari, P. Yin,T. Duke, A. Levskaya, S. Ghemawat, S. Dev, H. Michalewski,X. Garcia, V. Misra, K. Robinson, L. Fedus, D. Zhou, D. Ippolito,D. Luan, H. Lim, B. Zoph, A. Spiridonov, R. Sepassi, D. Dohan,S. Agrawal, M. Omernick, A. M. Dai, T. S. Pillai, M. Pellat,A. Lewkowycz, E. Moreira, R. Child, O. Polozov, K. Lee, Z. Zhou,X. Wang, B. Saeta, M. Diaz, O. Firat, M. Catasta, J. Wei, K. Meier-Hellstern, D. Eck, J. Dean, S. Petrov, and N. Fiedel, “Palm: Scalinglanguage modeling with pathways,” J. Mach. Learn. Res., vol. 24,pp. 240:1–240:113, 2023.





[16] J. Bai, S. Bai, Y. Chu, Z. Cui, K. Dang, X. Deng, Y. Fan, W. Ge,Y. Han, F. Huang, B. Hui, L. Ji, M. Li, J. Lin, R. Lin, D. Liu,G. Liu, C. Lu, K. Lu, J. Ma, R. Men, X. Ren, X. Ren, C. Tan, S. Tan,J. Tu, P. Wang, S. Wang, W. Wang, S. Wu, B. Xu, J. Xu, A. Yang,H. Yang, J. Yang, S. Yang, Y. Yao, B. Yu, H. Yuan, Z. Yuan, J. Zhang,X. Zhang, Y. Zhang, Z. Zhang, C. Zhou, J. Zhou, X. Zhou, andT. Zhu, “Qwen technical report,” CoRR, vol. abs/2309.16609,2023.





[17] Z. Zhang, X. Han, Z. Liu, X. Jiang, M. Sun, and Q. Liu, “ERNIE:enhanced language representation with informative entities,” inProceedings of the 57th Conference of the Association for Compu-tational Linguistics, ACL 2019, Florence, Italy, July 28- August 2,2019, Volume 1: Long Papers, A. Korhonen, D. R. Traum, and





L. Marquez, Eds. Association for Computational Linguistics, `2019, pp. 1441–1451.





[18] H. Touvron, L. Martin, K. R. Stone, P. Albert, A. Almahairi,Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale,D. M. Bikel, L. Blecher, C. C. Ferrer, M. Chen, G. Cucurull,D. Esiobu, J. Fernandes, J. Fu, W. Fu, B. Fuller, C. Gao,V. Goswami, N. Goyal, A. S. Hartshorn, S. Hosseini, R. Hou,H. Inan, M. Kardas, V. Kerkez, M. Khabsa, I. M. Kloumann,A. V. Korenev, P. S. Koura, M.-A. Lachaux, T. Lavril, J. Lee,D. Liskovich, Y. Lu, Y. Mao, X. Martinet, T. Mihaylov,P. Mishra, I. Molybog, Y. Nie, A. Poulton, J. Reizenstein,R. Rungta, K. Saladi, A. Schelten, R. Silva, E. M. Smith,R. Subramanian, X. Tan, B. Tang, R. Taylor, A. Williams,J. X. Kuan, P. Xu, Z. Yan, I. Zarov, Y. Zhang, A. Fan,M. Kambadur, S. Narang, A. Rodriguez, R. Stojnic, S. Edunov,and T. Scialom, “Llama 2: Open foundation and fine-tuned chatmodels,” ArXiv, vol. abs/2307.09288, 2023. [Online]. Available:https://api.semanticscholar.org/CorpusID:259950998





[19] J. Wei, Y. Tay, R. Bommasani, C. Raffel, B. Zoph, S. Borgeaud,D. Yogatama, M. Bosma, D. Zhou, D. Metzler, E. H. Chi,T. Hashimoto, O. Vinyals, P. Liang, J. Dean, and W. Fedus,“Emergent abilities of large language models,” Trans. Mach. Learn.Res., vol. 2022, 2022.





[20] G. Yang, D. Lo, R. Mullins, and Y. Zhao, “Dynamic stashingquantization for efficient transformer training,” arXiv preprintarXiv:2303.05295, 2023.





[21] Z. Yao, X. Wu, C. Li, S. Youn, and Y. He, “Zeroquant-v2: Exploringpost-training quantization in llms from comprehensive study tolow rank compensation,” 2023.





[22] Y. Bondarenko, M. Nagel, and T. Blankevoort, “Understandingand overcoming the challenges of efficient transformer quantiza-tion,” arXiv preprint arXiv:2109.12948, 2021.





[23] O. Zafrir, G. Boudoukh, P. Izsak, and M. Wasserblat, “Q8bert:Quantized 8bit bert,” in 2019 Fifth Workshop on Energy Effi-cient Machine Learning and Cognitive Computing-NeurIPS Edition(EMC2-NIPS). IEEE, 2019, pp. 36–39.





[24] B. Jacob, S. Kligys, B. Chen, M. Zhu, M. Tang, A. Howard,H. Adam, and D. Kalenichenko, “Quantization and training ofneural networks for efficient integer-arithmetic-only inference,”in Proceedings of the IEEE conference on computer vision and patternrecognition, 2018, pp. 2704–2713.





[25] S. Shen, Z. Dong, J. Ye, L. Ma, Z. Yao, A. Gholami, M. W. Ma-honey, and K. Keutzer, “Q-bert: Hessian based ultra low precisionquantization of bert,” in Proceedings of the AAAI Conference onArtificial Intelligence, vol. 34, no. 05, 2020, pp. 8815–8821.





[26] T. Piao, I. Cho, and U. Kang, “Sensimix: Sensitivity-aware 8-bit index & 1-bit value mixed precision quantization for bertcompression,” PloS one, vol. 17, no. 4, p. e0265621, 2022.





[27] W. Zhang, L. Hou, Y. Yin, L. Shang, X. Chen, X. Jiang, and Q. Liu,“Ternarybert: Distillation-aware ultra-low bit bert,” in Proceedingsof the 2020 Conference on Empirical Methods in Natural LanguageProcessing (EMNLP), 2020, pp. 509–521.





[28] H. Qin, Y. Ding, M. Zhang, Y. Qinghua, A. Liu, Q. Dang, Z. Liu,and X. Liu, “Bibert: Accurate fully binarized bert,” in InternationalConference on Learning Representations, 2021.





[29] C. Zhao, T. Hua, Y. Shen, Q. Lou, and H. Jin, “Automaticmixed-precision quantization search of bert,” arXiv preprintarXiv:2112.14938, 2021.





[30] Z. Zhao, Y. Liu, L. Chen, Q. Liu, R. Ma, and K. Yu, “Aninvestigation on different underlying quantization schemes forpre-trained language models,” in Natural Language Processingand Chinese Computing: 9th CCF International Conference, NLPCC2020, Zhengzhou, China, October 14–18, 2020, Proceedings, Part I 9.Springer, 2020, pp. 359–371.





[31] B. Wang, Y. Ren, L. Shang, X. Jiang, and Q. Liu, “Exploring ex-treme parameter compression for pre-trained language models,”in International Conference on Learning Representations, 2021.





[32] H. Bai, W. Zhang, L. Hou, L. Shang, J. Jin, X. Jiang, Q. Liu, M. Lyu,and I. King, “Binarybert: Pushing the limit of bert quantization,”in Proceedings of the 59th Annual Meeting of the Association forComputational Linguistics and the 11th International Joint Conferenceon Natural Language Processing (Volume 1: Long Papers), 2021, pp.4334–4348.





[33] A. H. Zadeh, I. Edo, O. M. Awad, and A. Moshovos, “Gobo:Quantizing attention-based nlp models for low latency and en-ergy efficient inference,” in 2020 53rd Annual IEEE/ACM Interna-





tional Symposium on Microarchitecture (MICRO). IEEE, 2020, pp.811–824.





[34] S. Kim, A. Gholami, Z. Yao, M. W. Mahoney, and K. Keutzer, “I-bert: Integer-only bert quantization,” in International conference onmachine learning. PMLR, 2021, pp. 5506–5518.





[35] S. Dai, R. Venkatesan, M. Ren, B. Zimmer, W. Dally, andB. Khailany, “Vs-quant: Per-vector scaled quantization for ac-curate low-precision neural network inference,” Proceedings ofMachine Learning and Systems, vol. 3, pp. 873–884, 2021.





[36] T. Li, Y. E. Mesbahi, I. Kobyzev, A. Rashid, A. Mahmud, N. An-churi, H. Hajimolahoseini, Y. Liu, and M. Rezagholizadeh, “Ashort study on compressing decoder-based language models,”arXiv preprint arXiv:2110.08460, 2021.





[37] C. Tao, L. Hou, W. Zhang, L. Shang, X. Jiang, Q. Liu, P. Luo,and N. Wong, “Compression of generative pre-trained languagemodels via quantization,” in Proceedings of the 60th Annual Meet-ing of the Association for Computational Linguistics (Volume 1: LongPapers), 2022, pp. 4821–4836.





[38] Z. Li, Z. Wang, M. Tan, R. Nallapati, P. Bhatia, A. Arnold,B. Xiang, and D. Roth, “Dq-bart: Efficient sequence-to-sequencemodel via joint distillation and quantization,” in Proceedings of the60th Annual Meeting of the Association for Computational Linguistics(Volume 2: Short Papers), 2022, pp. 203–211.





[39] A. Gholami, S. Kim, Z. Dong, Z. Yao, M. W. Mahoney, andK. Keutzer, “A survey of quantization methods for efficient neu-ral network inference,” in Low-Power Computer Vision. Chapmanand Hall/CRC, 2022, pp. 291–326.





[40] C. Xu and J. McAuley, “A survey on model compression andacceleration for pretrained language models,” in Proceedings ofthe AAAI Conference on Artificial Intelligence, vol. 37, no. 9, 2023,pp. 10 566–10 575.





[41] J. Lin, J. Tang, H. Tang, S. Yang, X. Dang, and S. Han, “Awq:Activation-aware weight quantization for llm compression andacceleration,” arXiv preprint arXiv:2306.00978, 2023.





[42] E. Frantar, S. Ashkboos, T. Hoefler, and D. Alistarh, “Optq:Accurate quantization for generative pre-trained transformers,”in The Eleventh International Conference on Learning Representations,2022.





[43] T. Dettmers, M. Lewis, Y. Belkada, and L. Zettlemoyer, “Llm. int8(): 8-bit matrix multiplication for transformers at scale,” arXivpreprint arXiv:2208.07339, 2022.





[44] Z. Yao, R. Yazdani Aminabadi, M. Zhang, X. Wu, C. Li, and Y. He,“Zeroquant: Efficient and affordable post-training quantizationfor large-scale transformers,” Advances in Neural Information Pro-cessing Systems, vol. 35, pp. 27 168–27 183, 2022.





[45] G. Xiao, J. Lin, M. Seznec, H. Wu, J. Demouth, and S. Han,“Smoothquant: Accurate and efficient post-training quantizationfor large language models,” in International Conference on MachineLearning. PMLR, 2023, pp. 38 087–38 099.





[46] Z. Liu, B. Oguz, C. Zhao, E. Chang, P. Stock, Y. Mehdad, Y. Shi,R. Krishnamoorthi, and V. Chandra, “Llm-qat: Data-free quanti-zation aware training for large language models,” arXiv preprintarXiv:2305.17888, 2023.





[47] Y. Chai, J. Gkountouras, G. G. Ko, D. Brooks, and G.-Y. Wei, “Int2.1: Towards fine-tunable quantized large language models witherror correction through low-rank adaptation,” arXiv preprintarXiv:2306.08162, 2023.





[48] T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer,“Qlora: Efficient finetuning of quantized llms,” arXiv preprintarXiv:2305.14314, 2023.





[49] S. Kim, C. Hooper, A. Gholami, Z. Dong, X. Li, S. Shen, M. W.Mahoney, and K. Keutzer, “Squeezellm: Dense-and-sparse quan-tization,” arXiv preprint arXiv:2306.07629, 2023.





[50] Y. J. Kim, R. Henry, R. Fahim, and H. H. Awadalla, “Finequant:Unlocking efficiency with fine-grained weight-only quantizationfor llms,” arXiv preprint arXiv:2308.09723, 2023.





[51] G. Park, B. Park, S. J. Kwon, B. Kim, Y. Lee, and D. Lee, “nuqmm:Quantized matmul for efficient inference of large-scale generativelanguage models,” arXiv preprint arXiv:2206.09557, 2022.





[52] M. Rastegari, V. Ordonez, J. Redmon, and A. Farhadi, “Xnor-net: Imagenet classification using binary convolutional neuralnetworks,” in European conference on computer vision. Springer,2016, pp. 525–542.





[53] T. Dettmers, M. Lewis, S. Shleifer, and L. Zettlemoyer, “8-bit op-timizers via block-wise quantization,” in International Conferenceon Learning Representations, 2021.





[54] J. H. Lee, J. Kim, S. J. Kwon, and D. Lee, “Flexround: Learnablerounding based on element-wise division for post-training quan-tization,” arXiv preprint arXiv:2306.00317, 2023.





[55] J. Chee, Y. Cai, V. Kuleshov, and C. De Sa, “Quip: 2-bit quanti-zation of large language models with guarantees,” arXiv preprintarXiv:2307.13304, 2023.





[56] E. Frantar and D. Alistarh, “Optimal brain compression: Aframework for accurate post-training quantization and pruning,”Advances in Neural Information Processing Systems, vol. 35, pp.4475–4488, 2022.





[57] C. Lee, J. Jin, T. Kim, H. Kim, and E. Park, “Owq: Lessons learnedfrom activation outliers for weight quantization in large languagemodels,” arXiv preprint arXiv:2306.02272, 2023.





[58] W. Cheng, W. Zhang, H. Shen, Y. Cai, X. He, and K. Lv, “Optimizeweight rounding via signed gradient descent for the quantizationof llms,” arXiv preprint arXiv:2309.05516, 2023.





[59] M. Nagel, R. A. Amjad, M. Van Baalen, C. Louizos, andT. Blankevoort, “Up or down? adaptive rounding for post-training quantization,” in International Conference on MachineLearning. PMLR, 2020, pp. 7197–7206.





[60] S. Merity, C. Xiong, J. Bradbury, and R. Socher, “Pointer sentinelmixture models,” in International Conference on Learning Represen-tations, 2016.





[61] C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena,Y. Zhou, W. Li, and P. J. Liu, “Exploring the limits of transferlearning with a unified text-to-text transformer,” The Journal ofMachine Learning Research, vol. 21, no. 1, pp. 5485–5551, 2020.





[62] Y. Li, Y. Yu, C. Liang, P. He, N. Karampatziakis, W. Chen, andT. Zhao, “Loftq: Lora-fine-tuning-aware quantization for largelanguage models,” arXiv preprint arXiv:2310.08659, 2023.





[63] Z. Yuan, L. Niu, J. Liu, W. Liu, X. Wang, Y. Shang, G. Sun, Q. Wu,J. Wu, and B. Wu, “Rptq: Reorder-based post-training quantiza-tion for large language models,” arXiv preprint arXiv:2304.01089,2023.





[64] Y. Zhang, L. Zhao, S. Cao, W. Wang, T. Cao, F. Yang, M. Yang,S. Zhang, and N. Xu, “Integer or floating point? new outlooks forlow-bit quantization on large language models,” arXiv preprintarXiv:2305.12356, 2023.





[65] X. Wu, Z. Yao, and Y. He, “Zeroquant-fp: A leap forward in llmspost-training w4a8 quantization using floating-point formats,”arXiv preprint arXiv:2307.09782, 2023.





[66] X. Wu, Z. Yao, and Y. H. Zeroquant-fp, “A leap forward in llmspost-training w4a8 quantization using floating-point formats,”arXiv preprint arXiv:2307.09782, 2023.





[67] X. Wei, Y. Zhang, X. Zhang, R. Gong, S. Zhang, Q. Zhang, F. Yu,and X. Liu, “Outlier suppression: Pushing the limit of low-bittransformer language models,” Advances in Neural InformationProcessing Systems, vol. 35, pp. 17 402–17 414, 2022.





[68] X. Wei, Y. Zhang, Y. Li, X. Zhang, R. Gong, J. Guo, and X. Liu,“Outlier suppression+: Accurate quantization of large languagemodels by equivalent and optimal shifting and scaling,” arXivpreprint arXiv:2304.09145, 2023.





[69] Q. Li, Y. Zhang, L. Li, P. Yao, B. Zhang, X. Chu, Y. Sun, L. Du, andY. Xie, “Fptq: Fine-grained post-training quantization for largelanguage models,” arXiv preprint arXiv:2308.15987, 2023.





[70] W. Shao, M. Chen, Z. Zhang, P. Xu, L. Zhao, Z. Li, K. Zhang,P. Gao, Y. Qiao, and P. Luo, “Omniquant: Omnidirectionallycalibrated quantization for large language models,” arXiv preprintarXiv:2308.13137, 2023.





[71] J. Liu, R. Gong, X. Wei, Z. Dong, J. Cai, and B. Zhuang, “Qllm: Ac-curate and efficient low-bitwidth quantization for large languagemodels,” arXiv preprint arXiv:2310.08041, 2023.





[72] E. Yvinec, A. Dapgony, M. Cord, and K. Bailly, “Rex: Data-free residual quantization error expansion,” arXiv preprintarXiv:2203.14645, 2022.





[73] C. Guo, J. Tang, W. Hu, J. Leng, C. Zhang, F. Yang, Y. Liu,M. Guo, and Y. Zhu, “Olive: Accelerating large language modelsvia hardware-friendly outlier-victim pair quantization,” in Pro-ceedings of the 50th Annual International Symposium on ComputerArchitecture, 2023, pp. 1–15.





[74] M. Kim, S. Lee, S. Hong, D.-S. Chang, and J. Choi, “Under-standing and improving knowledge distillation for quantization-aware training of large transformer encoders,” arXiv preprintarXiv:2211.11014, 2022.





[75] J. O. Neill and S. Dutta, “Self-distilled quantization: Achievinghigh compression rates in transformer-based language models,”arXiv preprint arXiv:2307.05972, 2023.





[76] W.-Y. Hua, B. Williams, and D. Shamsi, “Lacos-bloom: Low-rankadaptation with contrastive objective on 8 bits siamese-bloom,”arXiv preprint arXiv:2305.06404, 2023.





[77] A. Kaushal, T. Vaidhya, and I. Rish, “Lord: Low rank decomposi-tion of monolingual code llms for one-shot compression,” arXivpreprint arXiv:2309.14021, 2023.





[78] Y. Xu, L. Xie, X. Gu, X. Chen, H. Chang, H. Zhang, Z. Chen,X. Zhang, and Q. Tian, “Qa-lora: Quantization-aware low-rank adaptation of large language models,” arXiv preprintarXiv:2309.14717, 2023.





[79] S. J. Kwon, J. Kim, J. Bae, K. M. Yoo, J.-H. Kim, B. Park, B. Kim,J.-W. Ha, N. Sung, and D. Lee, “Alphatuning: Quantization-aware parameter-efficient adaptation of large-scale pre-trainedlanguage models,” arXiv preprint arXiv:2210.03858, 2022.





[80] J. Kim, J. H. Lee, S. Kim, J. Park, K. M. Yoo, S. J. Kwon, andD. Lee, “Memory-efficient fine-tuning of compressed large lan-guage models via sub-4-bit integer quantization,” arXiv preprintarXiv:2305.14152, 2023.





[81] M. Park, J. You, M. Nagel, and S. Chang, “Quadapter: Adapter forgpt-2 quantization,” in Findings of the Association for ComputationalLinguistics: EMNLP 2022, 2022, pp. 2510–2517.





[82] Z. Xu, Z. Liu, B. Chen, Y. Tang, J. Wang, K. Zhou, X. Hu, andA. Shrivastava, “Compress, then prompt: Improving accuracy-efficiency trade-off of llm inference with transferable prompt,”arXiv preprint arXiv:2305.11186, 2023.





[83] H. Shen, H. Meng, B. Dong, Z. Wang, O. Zafrir, Y. Ding, Y. Luo,H. Chang, Q. Gao, Z. Wang et al., “An efficient sparse inferencesoftware accelerator for transformer-based language models oncpus,” arXiv preprint arXiv:2306.16601, 2023.





[84] T. Pegolotti, E. Frantar, D. Alistarh, and M. Puschel, “Generat- ¨ing efficient kernels for quantized inference on large languagemodels,” in Workshop on Efficient Systems for Foundation Models@ICML2023, 2023.





[85] K. Wang, Z. Liu, Y. Lin, J. Lin, and S. Han, “Haq: Hardware-awareautomated quantization with mixed precision,” in Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition,2019, pp. 8612–8620.





[86] C. Yu, T. Chen, and Z. Gan, “Boost transformer-based languagemodels with gpu-friendly sparsity and quantization,” in Findingsof the Association for Computational Linguistics: ACL 2023, 2023, pp.218–235.





[87] Z. Lin, G. Qu, Q. Chen, X. Chen, Z. Chen, and K. Huang, “Push-ing large language models to the 6g edge: Vision, challenges, andopportunities,” arXiv preprint arXiv:2309.16739, 2023.





[88] M. W. U. Rahman, M. M. Abrar, H. G. Copening, S. Hariri,S. Shao, P. Satam, and S. Salehi, “Quantized transformer lan-guage model implementations on edge devices,” arXiv preprintarXiv:2310.03971, 2023.





[89] E. Kurtic, D. Kuznedelev, E. Frantar, M. Goin, and D. Alistarh,“Sparse finetuning for inference acceleration of large languagemodels,” arXiv preprint arXiv:2310.06927, 2023.





[90] B. Isik, H. Kumbong, W. Ning, X. Yao, S. Koyejo, and C. Zhang,“Gpt-zip: Deep compression of finetuned large language mod-els,” in Workshop on Efficient Systems for Foundation Models@ICML2023, 2023.





[91] X. Wei, S. Gonugondla, W. Ahmad, S. Wang, B. Ray, H. Qian,X. Li, V. Kumar, Z. Wang, Y. Tian et al., “Greener yet powerful:Taming large code generation models with quantization,” arXivpreprint arXiv:2303.05378, 2023.





[92] T. Hu, C. Meinel, and H. Yang, “Empirical evaluation of post-training quantization methods for language tasks,” arXiv preprintarXiv:2210.16621, 2022.





[93] T. Dettmers and L. Zettlemoyer, “The case for 4-bit precision: k-bit inference scaling laws,” in International Conference on MachineLearning. PMLR, 2023, pp. 7750–7774.





[94] P. Liu, Z. Liu, Z.-F. Gao, D. Gao, W. X. Zhao, Y. Li, B. Ding, and J.-R. Wen, “Do emergent abilities exist in quantized large languagemodels: An empirical study,” arXiv preprint arXiv:2307.08072,2023.





[95] Y. Bondarenko, M. Nagel, and T. Blankevoort, “Quantizabletransformers: Removing outliers by helping attention heads donothing,” arXiv preprint arXiv:2306.12929, 2023.





[96] A. Ahmadian, S. Dash, H. Chen, B. Venkitesh, S. Gou, P. Blunsom,A. Ust ¨ un, and S. Hooker, “Intriguing properties of quantization ¨at scale,” arXiv preprint arXiv:2305.19268, 2023.





[97] W. Wen, C. Wu, Y. Wang, Y. Chen, and H. Li, “Learning structuredsparsity in deep neural networks,” Advances in neural informationprocessing systems, vol. 29, 2016.





[98] S. Han, J. Pool, J. Tran, and W. Dally, “Learning both weightsand connections for efficient neural network,” Advances in neuralinformation processing systems, vol. 28, 2015.





[99] S. Narang, E. Undersander, and G. Diamos, “Block-sparse recur-rent neural networks,” arXiv preprint arXiv:1711.02782, 2017.





[100] M. A. Gordon, K. Duh, and N. Andrews, “Compressing bert:Studying the effects of weight pruning on transfer learning,”arXiv preprint arXiv:2002.08307, 2020.





[101] T. Chen, J. Frankle, S. Chang, S. Liu, Y. Zhang, Z. Wang, andM. Carbin, “The lottery ticket hypothesis for pre-trained bert net-works,” Advances in neural information processing systems, vol. 33,pp. 15 834–15 846, 2020.





[102] S. Prasanna, A. Rogers, and A. Rumshisky, “When bert plays thelottery, all tickets are winning,” arXiv preprint arXiv:2005.00561,2020.





[103] A. K. Jaiswal, S. Liu, T. Chen, Y. Ding, and Z. Wang, “Instantsoup: Cheap pruning ensembles in a single pass can draw lotterytickets from large models,” in International Conference on MachineLearning. PMLR, 2023, pp. 14 691–14 701.





[104] O. Zafrir, A. Larey, G. Boudoukh, H. Shen, and M. Wasserblat,“Prune once for all: Sparse pre-trained language models,” arXivpreprint arXiv:2111.05754, 2021.





[105] M. Zhu and S. Gupta, “To prune, or not to prune: exploringthe efficacy of pruning for model compression,” arXiv preprintarXiv:1710.01878, 2017.





[106] E. Kurtic and D. Alistarh, “Gmp*: Well-tuned global magnitudepruning can outperform most bert-pruning methods,” arXivpreprint arXiv:2210.06384, 2022.





[107] L. Yin, S. Liu, A. Jaiswal, S. Kundu, and Z. Wang, “Junk dna hy-pothesis: A task-centric angle of llm pre-trained weights throughsparsity,” arXiv preprint arXiv:2310.02277, 2023.





[108] V. Sanh, T. Wolf, and A. Rush, “Movement pruning: Adaptivesparsity by fine-tuning,” Advances in Neural Information ProcessingSystems, vol. 33, pp. 20 378–20 389, 2020.





[109] T. Jiang, D. Wang, F. Zhuang, R. Xie, and F. Xia, “Pruning pre-trained language models without fine-tuning,” arXiv preprintarXiv:2210.06210, 2022.





[110] S. Ren and K. Q. Zhu, “Low-rank prune-and-factorize for lan-guage model compression,” arXiv preprint arXiv:2306.14152, 2023.





[111] Q. Zhang, S. Zuo, C. Liang, A. Bukharin, P. He, W. Chen,and T. Zhao, “Platon: Pruning large transformer models withupper confidence bound of weight importance,” in InternationalConference on Machine Learning. PMLR, 2022, pp. 26 809–26 823.





[112] Y. Li, F. Luo, C. Tan, M. Wang, S. Huang, S. Li, and J. Bai,“Parameter-efficient sparsity for large language models fine-tuning,” arXiv preprint arXiv:2205.11005, 2022.





[113] M. Zhang, C. Shen, Z. Yang, L. Ou, X. Yu, B. Zhuang et al.,“Pruning meets low-rank parameter-efficient fine-tuning,” arXivpreprint arXiv:2305.18403, 2023.





[114] Y. LeCun, J. Denker, and S. Solla, “Optimal brain damage,”Advances in neural information processing systems, vol. 2, 1989.





[115] B. Hassibi, D. G. Stork, and G. J. Wolff, “Optimal brain surgeonand general network pruning,” in IEEE international conference onneural networks. IEEE, 1993, pp. 293–299.





[116] E. Kurtic, D. Campos, T. Nguyen, E. Frantar, M. Kurtz, B. Fineran,M. Goin, and D. Alistarh, “The optimal bert surgeon: Scalable andaccurate second-order pruning for large language models,” arXivpreprint arXiv:2203.07259, 2022.





[117] C. Louizos, M. Welling, and D. P. Kingma, “Learning sparseneural networks through l 0 regularization,” arXiv preprintarXiv:1712.01312, 2017.





[118] F.-M. Guo, S. Liu, F. S. Mungall, X. Lin, and Y. Wang, “Reweightedproximal pruning for large-scale language representation,” arXivpreprint arXiv:1909.12486, 2019.





[119] A. Mishra, J. A. Latorre, J. Pool, D. Stosic, D. Stosic, G. Venkatesh,C. Yu, and P. Micikevicius, “Accelerating sparse deep neuralnetworks,” arXiv preprint arXiv:2104.08378, 2021.





[120] A. Zhou, Y. Ma, J. Zhu, J. Liu, Z. Zhang, K. Yuan, W. Sun,and H. Li, “Learning n: m fine-grained structured sparse neuralnetworks from scratch,” arXiv preprint arXiv:2102.04010, 2021.





[121] O. Nordstrom, “Unstructured pruning of pre-trained language ¨models tuned for sentiment classification.” 2022.





[122] B. Cui, Y. Li, and Z. Zhang, “Joint structured pruning and denseknowledge distillation for efficient transformer model compres-sion,” Neurocomputing, vol. 458, pp. 56–69, 2021.





[123] B. Li, Z. Kong, T. Zhang, J. Li, Z. Li, H. Liu, and C. Ding, “Effi-cient transformer-based large scale language representations us-ing hardware-friendly block structured pruning,” arXiv preprintarXiv:2009.08065, 2020.





[124] P. Michel, O. Levy, and G. Neubig, “Are sixteen heads reallybetter than one?” Advances in neural information processing systems,vol. 32, 2019.





[125] J. Li, R. Cotterell, and M. Sachan, “Differentiable subset pruningof transformer heads,” Transactions of the Association for Computa-tional Linguistics, vol. 9, pp. 1442–1459, 2021.





[126] Z. Yang, Y. Cui, X. Yao, and S. Wang, “Gradient-based intra-attention pruning on pre-trained language models,” arXivpreprint arXiv:2212.07634, 2022.





[127] G. Wang, Q. Cao, J. Yang, and Y. Sun, “Task-oriented memory-efficient pruning-adapter,” arXiv preprint arXiv:2303.14704, 2023.





[128] C. J. Maddison, A. Mnih, and Y. W. Teh, “The concrete distri-bution: A continuous relaxation of discrete random variables,”arXiv preprint arXiv:1611.00712, 2016.





[129] E. Voita, D. Talbot, F. Moiseev, R. Sennrich, and I. Titov, “Ana-lyzing multi-head self-attention: Specialized heads do the heavylifting, the rest can be pruned,” arXiv preprint arXiv:1905.09418,2019.





[130] F. Lagunas, E. Charlaix, V. Sanh, and A. M. Rush, “Block pruningfor faster transformers,” arXiv preprint arXiv:2109.04838, 2021.





[131] R. Xu, F. Luo, C. Wang, B. Chang, J. Huang, S. Huang, andF. Huang, “From dense to sparse: Contrastive pruning for betterpre-trained language model compression,” in Proceedings of theAAAI Conference on Artificial Intelligence, vol. 36, no. 10, 2022, pp.11 547–11 555.





[132] Z. Liu, F. Li, G. Li, and J. Cheng, “Ebert: Efficient bert inferencewith dynamic structured pruning,” in Findings of the Associationfor Computational Linguistics: ACL-IJCNLP 2021, 2021, pp. 4814–4823.





[133] A. Khetan and Z. Karnin, “schubert: Optimizing elements ofbert,” arXiv preprint arXiv:2005.06628, 2020.





[134] E. Kurtic, E. Frantar, and D. Alistarh, “Ziplm: Hardware-aware structured pruning of language models,” arXiv preprintarXiv:2302.04089, 2023.





[135] A. Klein, J. Golebiowski, X. Ma, V. Perrone, and C. Archambeau,“Structural pruning of large language models via neural archi-tecture search,” in AutoML Conference 2023 (Workshop), 2023.





[136] S. Park, H. Choi, and U. Kang, “Knowledge-preserving prun-ing for pre-trained language models without retraining,” arXivpreprint arXiv:2308.03449, 2023.





[137] Y. Li, Y. Yu, Q. Zhang, C. Liang, P. He, W. Chen, and T. Zhao,“Losparse: Structured compression of large language modelsbased on low-rank and sparse approximation,” arXiv preprintarXiv:2306.11222, 2023.





[138] M. Santacroce, Z. Wen, Y. Shen, and Y. Li, “What matters inthe structured pruning of generative language models?” arXivpreprint arXiv:2302.03773, 2023.





[139] N. Yang, Y. Jang, H. Lee, S. Jeong, and K. Jung, “Task-specificcompression for multi-task language models using attribution-based pruning,” in Findings of the Association for ComputationalLinguistics: EACL 2023, 2023, pp. 582–592.





[140] J. McCarley, R. Chakravarti, and A. Sil, “Structured prun-ing of a bert-based question answering model,” arXiv preprintarXiv:1910.06360, 2019.





[141] Z. Wang, J. Wohlwend, and T. Lei, “Structured pruning of largelanguage models,” arXiv preprint arXiv:1910.04732, 2019.





[142] M. Xia, Z. Zhong, and D. Chen, “Structured pruning learnscompact and accurate models,” arXiv preprint arXiv:2204.00408,2022.





[143] C. Tao, L. Hou, H. Bai, J. Wei, X. Jiang, Q. Liu, P. Luo, andN. Wong, “Structured pruning for efficient generative pre-trainedlanguage models,” in Findings of the Association for ComputationalLinguistics: ACL 2023, 2023, pp. 10 880–10 895.





[144] A. Fan, E. Grave, and A. Joulin, “Reducing transformerdepth on demand with structured dropout,” arXiv preprintarXiv:1909.11556, 2019.





[145] M. Zhang and Y. He, “Accelerating training of transformer-basedlanguage models with progressive layer dropping,” Advances inNeural Information Processing Systems, vol. 33, pp. 14 011–14 023,2020.





[146] H. Sajjad, F. Dalvi, N. Durrani, and P. Nakov, “On the effect ofdropping layers of pre-trained transformer models,” ComputerSpeech & Language, vol. 77, p. 101429, 2023.





[147] S. Goyal, A. R. Choudhury, S. Raje, V. Chakaravarthy, Y. Sabhar-wal, and A. Verma, “Power-bert: Accelerating bert inference viaprogressive word-vector elimination,” in International Conferenceon Machine Learning. PMLR, 2020, pp. 3690–3699.





[148] S. Kim, S. Shen, D. Thorsley, A. Gholami, W. Kwon, J. Hassoun,and K. Keutzer, “Learned token pruning for transformers,” inProceedings of the 28th ACM SIGKDD Conference on KnowledgeDiscovery and Data Mining, 2022, pp. 784–794.





[149] H. Wang, Z. Zhang, and S. Han, “Spatten: Efficient sparse at-tention architecture with cascade token and head pruning,” in2021 IEEE International Symposium on High-Performance ComputerArchitecture (HPCA). IEEE, 2021, pp. 97–110.





[150] Z. Lin, J. Z. Liu, Z. Yang, N. Hua, and D. Roth, “Pruning redun-dant mappings in transformer models via spectral-normalizedidentity prior,” arXiv preprint arXiv:2010.01791, 2020.





[151] M. Sun, Z. Liu, A. Bair, and J. Z. Kolter, “A simple and effectivepruning approach for large language models,” arXiv preprintarXiv:2306.11695, 2023.





[152] Y. Zhang, H. Bai, H. Lin, J. Zhao, L. Hou, and C. V. Cannistraci,“An efficient plug-and-play post-training pruning strategy inlarge language models,” 2023.





[153] Y. Li, L. Niu, X. Zhang, K. Liu, J. Zhu, and Z. Kang, “E-sparse:Boosting the large language model inference through entropy-based n: M sparsity,” arXiv preprint arXiv:2310.15929, 2023.





[154] E. Frantar and D. Alistarh, “Sparsegpt: Massive language modelscan be accurately pruned in one-shot,” 2023.





[155] H. Shao, B. Liu, and Y. Qian, “One-shot sensitivity-aware mixedsparsity pruning for large language models,” arXiv preprintarXiv:2310.09499, 2023.





[156] R. J. Das, L. Ma, and Z. Shen, “Beyond size: How gradientsshape pruning decisions in large language models,” arXiv preprintarXiv:2311.04902, 2023.





[157] Anonymous, “Pushing gradient towards zero: A novel pruningmethod for large language models,” 2024. [Online]. Available:https://openreview.net/forum?id=IU4L7wiwxw





[158] Y. An, X. Zhao, T. Yu, M. Tang, and J. Wang, “Fluctuation-basedadaptive structured pruning for large language models,” arXivpreprint arXiv:2312.11983, 2023.





[159] S. Ashkboos, M. L. Croci, M. G. d. Nascimento, T. Hoefler,and J. Hensman, “Slicegpt: Compress large language modelsby deleting rows and columns,” arXiv preprint arXiv:2401.15024,2024.





[160] X. Ma, G. Fang, and X. Wang, “Llm-pruner: On the struc-tural pruning of large language models,” arXiv preprintarXiv:2305.11627, 2023.





[161] T. Chen, T. Ding, B. Yadav, I. Zharkov, and L. Liang, “Lorashear:Efficient large language model structured pruning and knowl-edge recovery,” arXiv preprint arXiv:2310.18356, 2023.





[162] B. Zhao, H. Hajishirzi, and Q. Cao, “Apt: Adaptive pruningand tuning pretrained language models for efficient training andinference,” arXiv preprint arXiv:2401.12200, 2024.





[163] M. Xia, T. Gao, Z. Zeng, and D. Chen, “Sheared llama: Accelerat-ing language model pre-training via structured pruning,” arXivpreprint arXiv:2310.06694, 2023.





[164] S. Guo, J. Xu, L. L. Zhang, and M. Yang, “Compresso: Structuredpruning with collaborative prompting learns compact large lan-guage models,” arXiv preprint arXiv:2310.05015, 2023.





[165] T. F. van der Ouderaa, M. Nagel, M. van Baalen, Y. M.Asano, and T. Blankevoort, “The llm surgeon,” arXiv preprintarXiv:2312.17244, 2023.





[166] M. Williams and N. Aletras, “How does calibration data affectthe post-training pruning and quantization of large languagemodels?” arXiv preprint arXiv:2311.09755, 2023.





[167] M. Zimmer, M. Andoni, C. Spiegel, and S. Pokutta, “Perp: Re-thinking the prune-retrain paradigm in the era of llms,” arXivpreprint arXiv:2312.15230, 2023.





[168] S. Gholami and M. Omar, “Can pruning make large languagemodels more efficient?” arXiv preprint arXiv:2310.04573, 2023.





[169] T. Valicenti, J. Vidal, and R. Patnaik, “Mini-gpts: Efficient largelanguage models through contextual pruning,” arXiv preprintarXiv:2312.12682, 2023.





[170] Y. Ji, Y. Cao, and J. Liu, “Pruning large language models viaaccuracy predictor,” arXiv preprint arXiv:2309.09507, 2023.





[171] Anonymous, “Outlier weighed layerwise sparsity (OWL): Amissing secret sauce for pruning LLMs to high sparsity,” inSubmitted to The Twelfth International Conference on LearningRepresentations, 2023, under review. [Online]. Available: https://openreview.net/forum?id=pOBvr1PxFd





[172] — —, “BESA: Pruning large language models with blockwiseparameter-efficient sparsity allocation,” in The TwelfthInternational Conference on Learning Representations, 2024. [Online].Available: https://openreview.net/forum?id=gC6JTEU3jl





[173] A. Syed, P. H. Guo, and V. Sundarapandiyan, “Prune and tune:Improving efficient pruning techniques for massive languagemodels,” 2023.





[174] Y. Zhang, L. Zhao, M. Lin, Y. Sun, Y. Yao, X. Han, J. Tanner, S. Liu,and R. Ji, “Dynamic sparse no training: Training-free fine-tuningfor sparse llms,” arXiv preprint arXiv:2310.08915, 2023.





[175] V. Boza, “Fast and optimal weight update for pruned large ˇlanguage models,” 2024.





[176] H. Xia, Z. Zheng, Y. Li, D. Zhuang, Z. Zhou, X. Qiu, Y. Li, W. Lin,and S. L. Song, “Flash-llm: Enabling cost-effective and highly-efficient large generative model inference with unstructured spar-sity,” arXiv preprint arXiv:2309.10285, 2023.





[177] V. Srinivasan, D. Gandhi, U. Thakker, and R. Prabhakar, “Traininglarge language models efficiently with sparsity and dataflow,”arXiv preprint arXiv:2304.05511, 2023.





[178] G. Hinton, O. Vinyals, and J. Dean, “Distilling the knowledge ina neural network,” arXiv preprint arXiv:1503.02531, 2015.





[179] Y. Zhang, T. Xiang, T. M. Hospedales, and H. Lu, “Deep mutuallearning,” in Proceedings of the IEEE conference on computer visionand pattern recognition, 2018, pp. 4320–4328.





[180] A. Romero, N. Ballas, S. E. Kahou, A. Chassang, C. Gatta, andY. Bengio, “Fitnets: Hints for thin deep nets,” arXiv preprintarXiv:1412.6550, 2014.





[181] J. Yim, D. Joo, J.-H. Bae, and J. Kim, “A gift from knowledgedistillation: Fast optimization, network minimization andtransfer learning,” 2017 IEEE Conference on Computer Visionand Pattern Recognition (CVPR), pp. 7130–7138, 2017. [Online].Available: https://api.semanticscholar.org/CorpusID:206596723





[182] R. Tang, Y. Lu, L. Liu, L. Mou, O. Vechtomova, and J. Lin,“Distilling task-specific knowledge from bert into simple neuralnetworks,” arXiv preprint arXiv:1903.12136, 2019.





[183] S. Sun, Y. Cheng, Z. Gan, and J. Liu, “Patient knowledge distilla-tion for bert model compression,” arXiv preprint arXiv:1908.09355,2019.





[184] L. Hou, Z. Huang, L. Shang, X. Jiang, X. Chen, and Q. Liu, “Dyn-abert: Dynamic bert with adaptive width and depth,” Advancesin Neural Information Processing Systems, vol. 33, pp. 9782–9793,2020.





[185] W. Zhou, C. Xu, and J. McAuley, “Bert learns to teach: Knowledgedistillation with meta learning,” arXiv preprint arXiv:2106.04570,2021.





[186] S. Wu, H. Chen, X. Quan, Q. Wang, and R. Wang, “Ad-kd:Attribution-driven knowledge distillation for language modelcompression,” arXiv preprint arXiv:2305.10010, 2023.





[187] D. Chen, Y. Li, M. Qiu, Z. Wang, B. Li, B. Ding, H. Deng, J. Huang,W. Lin, and J. Zhou, “Adabert: Task-adaptive bert compressionwith differentiable neural architecture search,” arXiv preprintarXiv:2001.04246, 2020.





[188] K. J. Liang, W. Hao, D. Shen, Y. Zhou, W. Chen, C. Chen, andL. Carin, “Mixkd: Towards efficient distillation of large-scalelanguage models,” arXiv preprint arXiv:2011.00593, 2020.





[189] H. Pan, C. Wang, M. Qiu, Y. Zhang, Y. Li, and J. Huang, “Meta-kd:A meta knowledge distillation framework for language modelcompression across domains,” arXiv preprint arXiv:2012.01266,2020.





[190] J. Zhang, A. Muhamed, A. Anantharaman, G. Wang, C. Chen,K. Zhong, Q. Cui, Y. Xu, B. Zeng, T. M. Chilimbi, andY. Chen, “Reaugkd: Retrieval-augmented knowledge distillationfor pre-trained language models,” in Annual Meeting of theAssociation for Computational Linguistics, 2023. [Online]. Available:https://api.semanticscholar.org/CorpusID:259370551





[191] V. Sanh, L. Debut, J. Chaumond, and T. Wolf, “Distilbert, adistilled version of bert: smaller, faster, cheaper and lighter,”arXiv preprint arXiv:1910.01108, 2019.





[192] W. Wang, F. Wei, L. Dong, H. Bao, N. Yang, and M. Zhou,“Minilm: Deep self-attention distillation for task-agnostic com-pression of pre-trained transformers,” Advances in Neural Infor-mation Processing Systems, vol. 33, pp. 5776–5788, 2020.





[193] Z. Sun, H. Yu, X. Song, R. Liu, Y. Yang, and D. Zhou, “Mobilebert:a compact task-agnostic bert for resource-limited devices,” arXivpreprint arXiv:2004.02984, 2020.





[194] C. Liang, H. Jiang, Z. Li, X. Tang, B. Yin, and T. Zhao, “Homodis-til: Homotopic task-agnostic distillation of pre-trained transform-ers,” arXiv preprint arXiv:2302.09632, 2023.





[195] X. Jiao, Y. Yin, L. Shang, X. Jiang, X. Chen, L. Li, F. Wang, andQ. Liu, “Tinybert: Distilling bert for natural language under-standing,” arXiv preprint arXiv:1909.10351, 2019.





[196] C. Liang, S. Zuo, Q. Zhang, P. He, W. Chen, and T. Zhao, “Lessis more: Task-aware layer-wise distillation for language modelcompression,” in International Conference on Machine Learning.PMLR, 2023, pp. 20 852–20 867.





[197] I. Turc, M.-W. Chang, K. Lee, and K. Toutanova, “Well-readstudents learn better: On the importance of pre-training compactmodels,” arXiv preprint arXiv:1908.08962, 2019.





[198] H. Zhang, M. Cisse, Y. N. Dauphin, and D. Lopez-Paz,“mixup: Beyond empirical risk minimization,” arXiv preprintarXiv:1710.09412, 2017.





[199] S. Dasgupta, T. Cohn, and T. Baldwin, “Cost-effectivedistillation of large language models,” in Annual Meeting of theAssociation for Computational Linguistics, 2023. [Online]. Available:https://api.semanticscholar.org/CorpusID:259858962





[200] S. I. Mirzadeh, M. Farajtabar, A. Li, N. Levine, A. Matsukawa,and H. Ghasemzadeh, “Improved knowledge distillation viateacher assistant,” in Proceedings of the AAAI conference on artificialintelligence, vol. 34, no. 04, 2020, pp. 5191–5198.





[201] Y. Wang, Y. Kordi, S. Mishra, A. Liu, N. A. Smith, D. Khashabi,and H. Hajishirzi, “Self-instruct: Aligning language model withself generated instructions,” arXiv preprint arXiv:2212.10560, 2022.





[202] B. Peng, C. Li, P. He, M. Galley, and J. Gao, “Instruction tuningwith gpt-4,” arXiv preprint arXiv:2304.03277, 2023.





[203] M. Wu, A. Waheed, C. Zhang, M. Abdul-Mageed, and A. F. Aji,“Lamini-lm: A diverse herd of distilled models from large-scaleinstructions,” arXiv preprint arXiv:2304.14402, 2023.





[204] Y. Jiang, C. Chan, M. Chen, and W. Wang, “Lion: Adversarialdistillation of closed-source large language model,” arXiv preprintarXiv:2305.12870, 2023.





[205] R. Taori, I. Gulrajani, T. Zhang, Y. Dubois, X. Li,C. Guestrin, P. Liang, and T. B. Hashimoto, “Stan-ford alpaca: An instruction-following llama model,”https://github.com/tatsu-lab/stanford alpaca, 2023.





[206] W.-L. Chiang, Z. Li, Z. Lin, Y. Sheng, Z. Wu, H. Zhang,L. Zheng, S. Zhuang, Y. Zhuang, J. E. Gonzalez, I. Stoica, andE. P. Xing, “Vicuna: An open-source chatbot impressing gpt-4with $9 0 \% ^ { * }$ chatgpt quality,” March 2023. [Online]. Available:https://lmsys.org/blog/2023-03-30-vicuna/





[207] Y. Anand, Z. Nussbaum, B. Duderstadt, B. Schmidt, and A. Mul-yar, “Gpt4all: Training an assistant-style chatbot with large scaledata distillation from gpt-3.5-turbo,” GitHub, 2023.





[208] H. Chen, A. Saha, S. Hoi, and S. Joty, “Personalised distillation:Empowering open-sourced llms with adaptive learning for codegeneration,” arXiv preprint arXiv:2310.18628, 2023.





[209] W. Zhou, S. Zhang, Y. Gu, M. Chen, and H. Poon, “Universalner:Targeted distillation from large language models for open namedentity recognition,” arXiv preprint arXiv:2308.03279, 2023.





[210] S. Li, J. Chen, Y. Shen, Z. Chen, X. Zhang, Z. Li, H. Wang, J. Qian,B. Peng, Y. Mao et al., “Explanations from large language modelsmake small reasoners better,” arXiv preprint arXiv:2210.06726,2022.





[211] L. C. Magister, J. Mallinson, J. Adamek, E. Malmi, and A. Sev-eryn, “Teaching small language models to reason,” arXiv preprintarXiv:2212.08410, 2022.





[212] C.-Y. Hsieh, C.-L. Li, C.-K. Yeh, H. Nakhost, Y. Fujii, A. Ratner,R. Krishna, C.-Y. Lee, and T. Pfister, “Distilling step-by-step!outperforming larger language models with less training dataand smaller model sizes,” arXiv preprint arXiv:2305.02301, 2023.





[213] S. Wadhwa, S. Amir, and B. C. Wallace, “Revisiting relationextraction in the era of large language models,” arXiv preprintarXiv:2305.05003, 2023.





[214] N. Ho, L. Schmid, and S.-Y. Yun, “Large language models arereasoning teachers,” arXiv preprint arXiv:2212.10071, 2022.





[215] K. Shridhar, A. Stolfo, and M. Sachan, “Distilling reasoningcapabilities into smaller language models,” in Findings of theAssociation for Computational Linguistics: ACL 2023, 2023, pp. 7059–7073.





[216] P. Wang, Z. Wang, Z. Li, Y. Gao, B. Yin, and X. Ren, “Scott:Self-consistent chain-of-thought distillation,” arXiv preprintarXiv:2305.01879, 2023.





[217] M. Kang, S. Lee, J. Baek, K. Kawaguchi, and S. J. Hwang,“Knowledge-augmented reasoning distillation for small lan-guage models in knowledge-intensive tasks,” arXiv preprintarXiv:2305.18395, 2023.





[218] Z. Jie and W. Lu, “Leveraging training data in few-shot prompt-ing for numerical reasoning,” arXiv preprint arXiv:2305.18170,2023.





[219] X. Zhu, B. Qi, K. Zhang, X. Long, and B. Zhou, “Pad: Program-aided distillation specializes large models in reasoning,” arXivpreprint arXiv:2305.13888, 2023.





[220] L. H. Li, J. Hessel, Y. Yu, X. Ren, K.-W. Chang, and Y. Choi,“Symbolic chain-of-thought distillation: Small models can also”think” step-by-step,” arXiv preprint arXiv:2306.14050, 2023.





[221] H. Chen, S. Wu, X. Quan, R. Wang, M. Yan, and J. Zhang, “Mcc-kd: Multi-cot consistent knowledge distillation,” arXiv preprintarXiv:2310.14747, 2023.





[222] H. Chae, Y. Song, K. T.-i. Ong, T. Kwon, M. Kim, Y. Yu, D. Lee,D. Kang, and J. Yeo, “Dialogue chain-of-thought distillationfor commonsense-aware conversational agents,” arXiv preprintarXiv:2310.09343, 2023.





[223] Z. Wang, S. Huang, Y. Liu, J. Wang, M. Song, Z. Zhang, H. Huang,F. Wei, W. Deng, F. Sun et al., “Democratizing reasoning ability:Tailored learning from large language model,” arXiv preprintarXiv:2310.13332, 2023.





[224] Y. Ma, H. Jiang, and C. Fan, “Sci-cot: Leveraging large languagemodels for enhanced knowledge distillation in small models forscientific qa,” arXiv preprint arXiv:2308.04679, 2023.





[225] Y. Fu, H. Peng, L. Ou, A. Sabharwal, and T. Khot, “Specializingsmaller language models towards multi-step reasoning,” arXivpreprint arXiv:2301.12726, 2023.





[226] Y. Huang, Y. Chen, Z. Yu, and K. McKeown, “In-context learningdistillation: Transferring few-shot learning ability of pre-trainedlanguage models,” arXiv preprint arXiv:2212.10670, 2022.





[227] L. Wang, N. Yang, and F. Wei, “Learning to retrieve in-context examples for large language models,” arXiv preprintarXiv:2307.07164, 2023.





[228] P. West, C. Bhagavatula, J. Hessel, J. D. Hwang, L. Jiang, R. L.Bras, X. Lu, S. Welleck, and Y. Choi, “Symbolic knowledge distil-lation: from general language models to commonsense models,”arXiv preprint arXiv:2110.07178, 2021.





[229] Z. Chen, Q. Gao, A. Bosselut, A. Sabharwal, and K. Richardson,“Disco: distilling counterfactuals with large language models,”in Proceedings of the 61st Annual Meeting of the Association forComputational Linguistics (Volume 1: Long Papers), 2023, pp. 5514–5528.





[230] Y. Gu, S. Zhang, N. Usuyama, Y. Woldesenbet, C. Wong, P. Sana-pathi, M. Wei, N. Valluri, E. Strandberg, T. Naumann et al.,“Distilling large language models for biomedical knowledgeextraction: A case study on adverse drug events,” arXiv preprintarXiv:2307.06439, 2023.





[231] G. Sahu, O. Vechtomova, D. Bahdanau, and I. H. Laradji,“Promptmix: A class boundary augmentation method for largelanguage model distillation,” arXiv preprint arXiv:2310.14192,2023.





[232] A. Gudibande, E. Wallace, C. Snell, X. Geng, H. Liu, P. Abbeel,S. Levine, and D. Song, “The false promise of imitating propri-etary llms,” arXiv preprint arXiv:2305.15717, 2023.





[233] Y. Gu, L. Dong, F. Wei, and M. Huang, “Knowledge distillationof large language models,” arXiv preprint arXiv:2306.08543, 2023.





[234] R. Agarwal, N. Vieillard, Y. Zhou, P. Stanczyk, S. Ramos, M. Geist,and O. Bachem, “Generalized knowledge distillation for auto-regressive language models,” arXiv preprint arXiv:2306.13649,2023.





[235] S. Padmanabhan, Y. Onoe, M. J. Zhang, G. Durrett, and E. Choi,“Propagating knowledge updates to lms through distillation,”arXiv preprint arXiv:2306.09306, 2023.





[236] M. Kim, S. Lee, J. Lee, S. Hong, D.-S. Chang, W. Sung, and J. Choi,“Token-scaled logit distillation for ternary weight generativelanguage models,” arXiv preprint arXiv:2308.06744, 2023.





[237] C. Zhang, D. Song, Z. Ye, and Y. Gao, “Towards the lawof capacity gap in distilling language models,” arXiv preprintarXiv:2311.07052, 2023.





[238] Z. Dai, Z. Yang, Y. Yang, J. Carbonell, Q. V. Le, and R. Salakhutdi-nov, “Transformer-xl: Attentive language models beyond a fixed-length context,” arXiv preprint arXiv:1901.02860, 2019.





[239] R. Child, S. Gray, A. Radford, and I. Sutskever, “Generat-ing long sequences with sparse transformers,” arXiv preprintarXiv:1904.10509, 2019.





[240] S. Sukhbaatar, E. Grave, P. Bojanowski, and A. Joulin, “Adaptiveattention span in transformers,” arXiv preprint arXiv:1905.07799,2019.





[241] G. M. Correia, V. Niculae, and A. F. Martins, “Adaptively sparsetransformers,” arXiv preprint arXiv:1909.00015, 2019.





[242] Z. Ye, Q. Guo, Q. Gan, X. Qiu, and Z. Zhang, “Bp-transformer:Modelling long-range context via binary partitioning,” arXivpreprint arXiv:1911.04070, 2019.





[243] J. Qiu, H. Ma, O. Levy, S. W.-t. Yih, S. Wang, and J. Tang,“Blockwise self-attention for long document understanding,”arXiv preprint arXiv:1911.02972, 2019.





[244] I. Beltagy, M. E. Peters, and A. Cohan, “Longformer: The long-document transformer,” arXiv preprint arXiv:2004.05150, 2020.





[245] N. Kitaev, Ł. Kaiser, and A. Levskaya, “Reformer: The efficienttransformer,” arXiv preprint arXiv:2001.04451, 2020.





[246] M. Zaheer, G. Guruganesh, K. A. Dubey, J. Ainslie, C. Alberti,S. Ontanon, P. Pham, A. Ravula, Q. Wang, L. Yang et al., “Bigbird: Transformers for longer sequences,” Advances in neuralinformation processing systems, vol. 33, pp. 17 283–17 297, 2020.





[247] Y. Tay, D. Bahri, L. Yang, D. Metzler, and D.-C. Juan, “Sparsesinkhorn attention,” in International Conference on Machine Learn-ing. PMLR, 2020, pp. 9438–9447.





[248] X. Li, Y. Meng, M. Zhou, Q. Han, F. Wu, and J. Li, “Sac:Accelerating and structuring self-attention via sparse adaptiveconnection,” Advances in Neural Information Processing Systems,vol. 33, pp. 16 997–17 008, 2020.





[249] H. Ren, H. Dai, Z. Dai, M. Yang, J. Leskovec, D. Schuurmans, andB. Dai, “Combiner: Full attention transformer with sparse com-putation cost,” Advances in Neural Information Processing Systems,vol. 34, pp. 22 470–22 482, 2021.





[250] A. Roy, M. Saffar, A. Vaswani, and D. Grangier, “Efficient content-based sparse attention with routing transformers,” Transactions ofthe Association for Computational Linguistics, vol. 9, pp. 53–68, 2021.





[251] A. Katharopoulos, A. Vyas, N. Pappas, and F. Fleuret, “Trans-formers are rnns: Fast autoregressive transformers with linearattention,” in International conference on machine learning. PMLR,2020, pp. 5156–5165.





[252] K. Choromanski, V. Likhosherstov, D. Dohan, X. Song, A. Gane,T. Sarlos, P. Hawkins, J. Davis, A. Mohiuddin, L. Kaiseret al., “Rethinking attention with performers,” arXiv preprintarXiv:2009.14794, 2020.





[253] Y. Xiong, Z. Zeng, R. Chakraborty, M. Tan, G. Fung, Y. Li, andV. Singh, “Nystromformer: A nystr ¨ om-based algorithm for ap- ¨proximating self-attention,” in Proceedings of the AAAI Conferenceon Artificial Intelligence, vol. 35, no. 16, 2021, pp. 14 138–14 148.





[254] W. Hua, Z. Dai, H. Liu, and Q. Le, “Transformer quality in lineartime,” in International Conference on Machine Learning. PMLR,2022, pp. 9099–9117.





[255] I. Han, R. Jarayam, A. Karbasi, V. Mirrokni, D. P. Woodruff,and A. Zandieh, “Hyperattention: Long-context attention in near-linear time,” arXiv preprint arXiv:2310.05869, 2023.





[256] Z. Shen, M. Zhang, H. Zhao, S. Yi, and H. Li, “Efficient atten-tion: Attention with linear complexities,” in Proceedings of theIEEE/CVF winter conference on applications of computer vision, 2021,pp. 3531–3539.





[257] A. Rahimi and B. Recht, “Random features for large-scale ker-nel machines,” Advances in neural information processing systems,vol. 20, 2007.





[258] S. Wang, B. Z. Li, M. Khabsa, H. Fang, and H. Ma, “Lin-former: Self-attention with linear complexity,” arXiv preprintarXiv:2006.04768, 2020.





[259] L. D. Lingle, “Transformer-vq: Linear-time transformers via vec-tor quantization,” arXiv preprint arXiv:2309.16354, 2023.





[260] T. Dao, D. Fu, S. Ermon, A. Rudra, and C. Re, “Flashattention: ´Fast and memory-efficient exact attention with io-awareness,”Advances in Neural Information Processing Systems, vol. 35, pp.16 344–16 359, 2022.





[261] T. Dao, “Flashattention-2: Faster attention with better parallelismand work partitioning,” arXiv preprint arXiv:2307.08691, 2023.





[262] H. Wang, Z. Wu, Z. Liu, H. Cai, L. Zhu, C. Gan, and S. Han, “Hat:Hardware-aware transformers for efficient natural language pro-cessing,” arXiv preprint arXiv:2005.14187, 2020.





[263] P. Ren, Y. Xiao, X. Chang, P.-Y. Huang, Z. Li, X. Chen, andX. Wang, “A comprehensive survey of neural architecture search:Challenges and solutions,” ACM Computing Surveys (CSUR),vol. 54, no. 4, pp. 1–34, 2021.





[264] Y. Liu, Y. Sun, B. Xue, M. Zhang, G. G. Yen, and K. C. Tan,“A survey on evolutionary neural architecture search,” IEEEtransactions on neural networks and learning systems, 2021.





[265] T. Elsken, J. H. Metzen, and F. Hutter, “Neural architecture search:A survey,” The Journal of Machine Learning Research, vol. 20, no. 1,pp. 1997–2017, 2019.





[266] A. Wan, X. Dai, P. Zhang, Z. He, Y. Tian, S. Xie, B. Wu, M. Yu,T. Xu, K. Chen et al., “Fbnetv2: Differentiable neural architecturesearch for spatial and channel dimensions,” in Proceedings of theIEEE/CVF conference on computer vision and pattern recognition,2020, pp. 12 965–12 974.





[267] B. Wu, X. Dai, P. Zhang, Y. Wang, F. Sun, Y. Wu, Y. Tian,P. Vajda, Y. Jia, and K. Keutzer, “Fbnet: Hardware-aware efficientconvnet design via differentiable neural architecture search,”in Proceedings of the IEEE/CVF conference on computer vision andpattern recognition, 2019, pp. 10 734–10 742.





[268] D. So, Q. Le, and C. Liang, “The evolved transformer,” in Interna-tional conference on machine learning. PMLR, 2019, pp. 5877–5886.





[269] Y. Zhao, L. Dong, Y. Shen, Z. Zhang, F. Wei, andW. Chen, “Memory-efficient differentiable transformer architec-ture search,” arXiv preprint arXiv:2105.14669, 2021.





[270] H. Liu, K. Simonyan, and Y. Yang, “Darts: Differentiable architec-ture search,” arXiv preprint arXiv:1806.09055, 2018.





[271] J. Kaplan, S. McCandlish, T. Henighan, T. B. Brown, B. Chess,R. Child, S. Gray, A. Radford, J. Wu, and D. Amodei,“Scaling laws for neural language models,” arXiv preprintarXiv:2001.08361, 2020.





[272] J. Hestness, S. Narang, N. Ardalani, G. Diamos, H. Jun,H. Kianinejad, M. M. A. Patwary, Y. Yang, and Y. Zhou,“Deep learning scaling is predictable, empirically,” arXiv preprintarXiv:1712.00409, 2017.





[273] C. Xu and J. McAuley, “A survey on dynamic neural networksfor natural language processing,” arXiv preprint arXiv:2202.07101,2022.





[274] Y.-S. Chuang, Y. Xie, H. Luo, Y. Kim, J. R. Glass, and P. He, “Dola:Decoding by contrasting layers improves factuality in largelanguage models,” ArXiv, vol. abs/2309.03883, 2023. [Online].Available: https://api.semanticscholar.org/CorpusID:261582463





[275] J. Xin, R. Tang, J. Lee, Y. Yu, and J. J. Lin, “Deebert: Dynamicearly exiting for accelerating bert inference,” in Annual Meetingof the Association for Computational Linguistics, 2020. [Online].Available: https://api.semanticscholar.org/CorpusID:216552850





[276] W. Liu, P. Zhou, Z. Zhao, Z. Wang, H. Deng, andQ. Ju, “Fastbert: a self-distilling bert with adaptive inferencetime,” ArXiv, vol. abs/2004.02178, 2020. [Online]. Available:https://api.semanticscholar.org/CorpusID:214802887





[277] S. Geng, P. Gao, Z. Fu, and Y. Zhang, “Romebert: Robust trainingof multi-exit bert,” ArXiv, vol. abs/2101.09755, 2021. [Online].Available: https://api.semanticscholar.org/CorpusID:231698881





[278] J. Wang, K. Chen, G. Chen, L. Shou, and J. McAuley, “Skipbert:Efficient inference with shallow layer skipping,” in AnnualMeeting of the Association for Computational Linguistics, 2022.[Online]. Available: https://api.semanticscholar.org/CorpusID:248780497





[279] W. Zhu, “Leebert: Learned early exit for bert with cross-level optimization,” in Annual Meeting of the Associationfor Computational Linguistics, 2021. [Online]. Available: https://api.semanticscholar.org/CorpusID:236459809





[280] J. Kong, J. Wang, L.-C. Yu, and X. Zhang, “Acceleratinginference for pretrained language models by unified multi-perspective early exiting,” in International Conference onComputational Linguistics, 2022. [Online]. Available: https://api.semanticscholar.org/CorpusID:252818912





[281] D. Ye, Y. Lin, Y. Huang, and M. Sun, “Tr-bert: Dynamic tokenreduction for accelerating bert inference,” in North AmericanChapter of the Association for Computational Linguistics, 2021.[Online]. Available: https://api.semanticscholar.org/CorpusID:235097557





[282] D. Zeng, N. Du, T. Wang, Y. Xu, T. Lei, Z. Chen,and C. Cui, “Learning to skip for language modeling,”





ArXiv, vol. abs/2311.15436, 2023. [Online]. Available: https://api.semanticscholar.org/CorpusID:265456419





[283] Y. Wang, K. Chen, H. Tan, and K. Guo, “Tabi: An efficient multi-level inference system for large language models,” Proceedingsof the Eighteenth European Conference on Computer Systems, 2023.[Online]. Available: https://api.semanticscholar.org/CorpusID:258508784





[284] L. Chen, M. A. Zaharia, and J. Y. Zou, “Frugalgpt: How touse large language models while reducing cost and improvingperformance,” ArXiv, vol. abs/2305.05176, 2023. [Online].Available: https://api.semanticscholar.org/CorpusID:258564349





[285] J. Zhang, R. Krishna, A. H. Awadallah, and C. Wang,“Ecoassistant: Using llm assistant more affordably andaccurately,” ArXiv, vol. abs/2310.03046, 2023. [Online]. Available:https://api.semanticscholar.org/CorpusID:263671677





[286] B. Zhu, Y. Sheng, L. Zheng, C. W. Barrett, M. I. Jordan, andJ. Jiao, “On optimal caching and model multiplexing for largemodel inference,” ArXiv, vol. abs/2306.02003, 2023. [Online].Available: https://api.semanticscholar.org/CorpusID:259075212





[287] M. Yue, J. Zhao, M. Zhang, L. Du, and Z. Yao, “Large languagemodel cascades with mixture of thoughts representationsfor cost-efficient reasoning,” ArXiv, vol. abs/2310.03094, 2023.[Online]. Available: https://api.semanticscholar.org/CorpusID:263671564





[288] D. Patel and G. Wong, “Gpt-4 architecture, infrastructure, train-ing dataset, costs, vision, moe,” 2023, https://www.semianalysis.com/p/gpt-4-architecture-infrastructure.





[289] A. Q. Jiang, A. Sablayrolles, A. Roux, A. Mensch, B. Savary,C. Bamford, D. S. Chaplot, D. d. l. Casas, E. B. Hanna, F. Bressandet al., “Mixtral of experts,” arXiv preprint arXiv:2401.04088, 2024.





[290] S. Rajbhandari, C. Li, Z. Yao, M. Zhang, R. Y. Aminabadi,A. A. Awan, J. Rasley, and Y. He, “Deepspeed-moe: Advanc-ing mixture-of-experts inference and training to power next-generation ai scale,” in International Conference on Machine Learn-ing. PMLR, 2022, pp. 18 332–18 346.





[291] D. Dai, C. Deng, C. Zhao, R. X. Xu, H. Gao, D. Chen,J. Li, W. Zeng, X. Yu, Y. Wu, Z. Xie, Y. K. Li, P. Huang,F. Luo, C. Ruan, Z. Sui, and W. Liang, “Deepseekmoe: Towardsultimate expert specialization in mixture-of-experts languagemodels,” ArXiv, vol. abs/2401.06066, 2024. [Online]. Available:https://api.semanticscholar.org/CorpusID:266933338





[292] N. Shazeer, A. Mirhoseini, K. Maziarz, A. Davis, Q. Le, G. Hinton,and J. Dean, “Outrageously large neural networks: The sparsely-gated mixture-of-experts layer,” arXiv preprint arXiv:1701.06538,2017.





[293] D. Lepikhin, H. Lee, Y. ${ \mathrm { X u } } ,$ D. Chen, O. Firat, Y. Huang,M. Krikun, N. Shazeer, and Z. Chen, “Gshard: Scaling giantmodels with conditional computation and automatic sharding,”arXiv preprint arXiv:2006.16668, 2020.





[294] W. Fedus, B. Zoph, and N. Shazeer, “Switch transformers: Scalingto trillion parameter models with simple and efficient sparsity,”The Journal of Machine Learning Research, vol. 23, no. 1, pp. 5232–5270, 2022.





[295] Y. Zhou, T. Lei, H. Liu, N. Du, Y. Huang, V. Zhao, A. M.Dai, Q. V. Le, J. Laudon et al., “Mixture-of-experts with expertchoice routing,” Advances in Neural Information Processing Systems,vol. 35, pp. 7103–7114, 2022.





[296] A. Yang, J. Lin, R. Men, C. Zhou, L. Jiang, X. Jia, A. Wang,J. Zhang, J. Wang, Y. Li et al., “M6-t: Exploring sparse expertmodels and beyond,” arXiv preprint arXiv:2105.15082, 2021.





[297] Y. Zhou, N. Du, Y. Huang, D. Peng, C. Lan, D. Huang, S. Shakeri,D. So, A. M. Dai, Y. Lu et al., “Brainformers: Trading simplicityfor efficiency,” in International Conference on Machine Learning.PMLR, 2023, pp. 42 531–42 542.





[298] R. A. Jacobs, M. I. Jordan, S. J. Nowlan, and G. E. Hinton,“Adaptive mixtures of local experts,” Neural computation, vol. 3,no. 1, pp. 79–87, 1991.





[299] M. I. Jordan and R. A. Jacobs, “Hierarchical mixtures of expertsand the em algorithm,” Neural computation, vol. 6, no. 2, pp. 181–214, 1994.





[300] A. Graves and A. Graves, “Long short-term memory,” Supervisedsequence labelling with recurrent neural networks, pp. 37–45, 2012.





[301] Z. Chi, L. Dong, S. Huang, D. Dai, S. Ma, B. Patra, S. Singhal,P. Bajaj, X. Song, X.-L. Mao et al., “On the representation collapseof sparse mixture of experts,” Advances in Neural InformationProcessing Systems, vol. 35, pp. 34 600–34 613, 2022.





[302] Y. Xie, S. Huang, T. Chen, and F. Wei, “Moec: Mixture of expertclusters,” in Proceedings of the AAAI Conference on Artificial Intelli-gence, vol. 37, no. 11, 2023, pp. 13 807–13 815.





[303] M. Lewis, S. Bhosale, T. Dettmers, N. Goyal, and L. Zettlemoyer,“Base layers: Simplifying training of large, sparse models,” inInternational Conference on Machine Learning. PMLR, 2021, pp.6265–6274.





[304] A. Clark, D. De Las Casas, A. Guy, A. Mensch, M. Paganini,J. Hoffmann, B. Damoc, B. Hechtman, T. Cai, S. Borgeaud et al.,“Unified scaling laws for routed language models,” in Interna-tional Conference on Machine Learning. PMLR, 2022, pp. 4057–4086.





[305] S. Roller, S. Sukhbaatar, J. Weston et al., “Hash layers for largesparse models,” Advances in Neural Information Processing Systems,vol. 34, pp. 17 555–17 566, 2021.





[306] C. N. dos Santos, J. Lee-Thorp, I. Noble, C.-C. Chang, andD. Uthus, “Memory augmented language models throughmixture of word experts,” ArXiv, vol. abs/2311.10768, 2023.[Online]. Available: https://api.semanticscholar.org/CorpusID:265295488





[307] X. Ren, P. Zhou, X. Meng, X. Huang, Y. Wang, W. Wang, P. Li,X. Zhang, A. V. Podolskiy, G. Arshinov, A. Bout, I. Piontkovskaya,J. Wei, X. Jiang, T. Su, Q. Liu, and J. Yao, “Pangu-σ: Towards trillion parameter language model with sparseheterogeneous computing,” ArXiv, vol. abs/2303.10845, 2023.[Online]. Available: https://api.semanticscholar.org/CorpusID:257666647





[308] J. Li, Z. Sun, X. He, L. Zeng, Y. Lin, E. Li, B. Zheng,R. Zhao, and X. Chen, “Locmoe: A low-overhead moe forlarge language model training,” 2024. [Online]. Available:https://api.semanticscholar.org/CorpusID:267212059





[309] S. Zuo, X. Liu, J. Jiao, Y. J. Kim, H. Hassan, R. Zhang, T. Zhao, andJ. Gao, “Taming sparsely activated transformer with stochasticexperts,” arXiv preprint arXiv:2110.04260, 2021.





[310] Y. J. Kim, A. A. Awan, A. Muzio, A. F. C. Salinas, L. Lu, A. Hendy,S. Rajbhandari, Y. He, and H. H. Awadalla, “Scalable and efficientmoe training for multitask multilingual models,” arXiv preprintarXiv:2109.10465, 2021.





[311] C. Riquelme, J. Puigcerver, B. Mustafa, M. Neumann, R. Jenatton,A. Susano Pinto, D. Keysers, and N. Houlsby, “Scaling visionwith sparse mixture of experts,” Advances in Neural InformationProcessing Systems, vol. 34, pp. 8583–8595, 2021.





[312] B. Zoph, I. Bello, S. Kumar, N. Du, Y. Huang, J. Dean, N. Shazeer,and W. Fedus, “St-moe: Designing stable and transferable sparseexpert models,” arXiv preprint arXiv:2202.08906, 2022.





[313] H. Hazimeh, Z. Zhao, A. Chowdhery, M. Sathiamoorthy, Y. Chen,R. Mazumder, L. Hong, and E. Chi, “Dselect-k: Differentiableselection in the mixture of experts with applications to multi-task learning,” Advances in Neural Information Processing Systems,vol. 34, pp. 29 335–29 347, 2021.





[314] N. Du, Y. Huang, A. M. Dai, S. Tong, D. Lepikhin, Y. Xu,M. Krikun, Y. Zhou, A. W. Yu, O. Firat et al., “Glam: Efficient scal-ing of language models with mixture-of-experts,” in InternationalConference on Machine Learning. PMLR, 2022, pp. 5547–5569.





[315] M. Artetxe, S. Bhosale, N. Goyal, T. Mihaylov, M. Ott, S. Shleifer,X. V. Lin, J. Du, S. Iyer, R. Pasunuru et al., “Efficient largescale language modeling with mixtures of experts,” arXiv preprintarXiv:2112.10684, 2021.





[316] D. Dai, L. Dong, S. Ma, B. Zheng, Z. Sui, B. Chang, and F. Wei,“Stablemoe: Stable routing strategy for mixture of experts,” arXivpreprint arXiv:2204.08396, 2022.





[317] M. R. Costa-jussa, J. Cross, O. C¸ elebi, M. Elbayad, K. Heafield, `K. Heffernan, E. Kalbassi, J. Lam, D. Licht, J. Maillard et al., “Nolanguage left behind: Scaling human-centered machine transla-tion,” arXiv preprint arXiv:2207.04672, 2022.





[318] M. Tan and Q. Le, “Efficientnet: Rethinking model scaling forconvolutional neural networks,” in International conference onmachine learning. PMLR, 2019, pp. 6105–6114.





[319] O. Press, N. A. Smith, and O. Levy, “Improving trans-former models by reordering their sublayers,” arXiv preprintarXiv:1911.03864, 2019.





[320] D. So, W. Manke, H. Liu, Z. Dai, N. Shazeer, and Q. V. Le, ´“Searching for efficient transformers for language modeling,”Advances in Neural Information Processing Systems, vol. 34, pp.6010–6022, 2021.





[321] B. Zoph and Q. V. Le, “Neural architecture search with reinforce-ment learning,” arXiv preprint arXiv:1611.01578, 2016.





[322] X. Nie, X. Miao, S. Cao, L. Ma, Q. Liu, J. Xue, Y. Miao, Y. Liu,Z. Yang, and B. Cui, “Evomoe: An evolutional mixture-of-expertstraining framework via dense-to-sparse gate,” arXiv preprintarXiv:2112.14397, 2021.





[323] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, andR. Salakhutdinov, “Dropout: a simple way to prevent neural net-works from overfitting,” The journal of machine learning research,vol. 15, no. 1, pp. 1929–1958, 2014.





[324] R. Liu, Y. J. Kim, A. Muzio, and H. Hassan, “Gating dropout:Communication-efficient regularization for sparsely activatedtransformers,” in International Conference on Machine Learning.PMLR, 2022, pp. 13 782–13 792.





[325] Z. Zhang, Y. Gu, X. Han, S. Chen, C. Xiao, Z. Sun, Y. Yao, F. Qi,J. Guan, P. Ke et al., “Cpm-2: Large-scale cost-effective pre-trainedlanguage models,” AI Open, vol. 2, pp. 216–224, 2021.





[326] F. Xue, Z. Zheng, Y. Fu, J. Ni, Z. Zheng, W. Zhou, and Y. You,“Openmoe: An early effort on open mixture-of-experts languagemodels,” 2024.





[327] T. Chen, S. Huang, Y. Xie, B. Jiao, D. Jiang, H. Zhou, J. Li,and F. Wei, “Task-specific expert pruning for sparse mixture-of-experts,” arXiv preprint arXiv:2206.00277, 2022.





[328] Z.-F. Gao, P. Liu, W. X. Zhao, Z.-Y. Lu, and J.-R. Wen, “Parameter-efficient mixture-of-experts architecture for pre-trained languagemodels,” arXiv preprint arXiv:2203.01104, 2022.





[329] S. Zuo, Q. Zhang, C. Liang, P. He, T. Zhao, and W. Chen,“Moebert: from bert to mixture-of-experts via importance-guidedadaptation,” arXiv preprint arXiv:2204.07675, 2022.





[330] P. Molchanov, A. Mallya, S. Tyree, I. Frosio, and J. Kautz, “Impor-tance estimation for neural network pruning,” in Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition,2019, pp. 11 264–11 272.





[331] Z. Zhang, Y. Lin, Z. Liu, P. Li, M. Sun, and J. Zhou, “Moefication:Transformer feed-forward layers are mixtures of experts,” arXivpreprint arXiv:2110.01786, 2021.





[332] R. Csord’as, K. Irie, and J. Schmidhuber, “Approximatingtwo-layer feedforward networks for efficient transformers,”ArXiv, vol. abs/2310.10837, 2023. [Online]. Available: https://api.semanticscholar.org/CorpusID:264172384





[333] R. Csord’as, P. Piekos, K. Irie, and J. Schmidhuber, “Switchhead:Accelerating transformers with mixture-of-experts attention,”ArXiv, vol. abs/2312.07987, 2023. [Online]. Available: https://api.semanticscholar.org/CorpusID:266191825





[334] I. O. Tolstikhin, N. Houlsby, A. Kolesnikov, L. Beyer, X. Zhai,T. Unterthiner, J. Yung, A. Steiner, D. Keysers, J. Uszkoreit et al.,“Mlp-mixer: An all-mlp architecture for vision,” Advances inneural information processing systems, vol. 34, pp. 24 261–24 272,2021.





[335] J. Lee-Thorp and J. Ainslie, “Sparse mixers: Combining moeand mixing to build a more efficient bert,” arXiv preprintarXiv:2205.12399, 2022.





[336] P. Yu, M. Artetxe, M. Ott, S. Shleifer, H. Gong, V. Stoyanov, andX. Li, “Efficient language modeling with sparse all-mlp,” arXivpreprint arXiv:2203.06850, 2022.





[337] Y. Wang, S. Agarwal, S. Mukherjee, X. Liu, J. Gao, A. H. Awadal-lah, and J. Gao, “Adamix: Mixture-of-adaptations for parameter-efficient model tuning,” arXiv preprint arXiv:2210.17451, 2022.





[338] N. Houlsby, A. Giurgiu, S. Jastrzebski, B. Morrone, Q. De Larous-silhe, A. Gesmundo, M. Attariyan, and S. Gelly, “Parameter-efficient transfer learning for nlp,” in International Conference onMachine Learning. PMLR, 2019, pp. 2790–2799.





[339] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang,and W. Chen, “Lora: Low-rank adaptation of large languagemodels,” arXiv preprint arXiv:2106.09685, 2021.





[340] S. Diao, T. Xu, R. Xu, J. Wang, and T. Zhang, “Mixture-of-domain-adapters: Decoupling and injecting domain knowledgeto pre-trained language models’ memories,” in Annual Meetingof the Association for Computational Linguistics, 2023. [Online].Available: https://api.semanticscholar.org/CorpusID:259108831





[341] R. Li, G. Murray, and G. Carenini, “Mixture-of-linguistic-experts adapters for improving and interpreting pre-trainedlanguage models,” in Conference on Empirical Methods inNatural Language Processing, 2023. [Online]. Available: https://api.semanticscholar.org/CorpusID:264487239





[342] Y. Zhu, N. Wichers, C.-C. Lin, X. Wang, T. Chen, L. Shu, H. Lu,C. Liu, L. Luo, J. Chen, and L. Meng, “Sira: Sparse mixture oflow rank adaptation,” ArXiv, vol. abs/2311.09179, 2023. [Online].Available: https://api.semanticscholar.org/CorpusID:265213347





[343] S. Dou, E. Zhou, Y. Liu, S. Gao, J. Zhao, W. Shen, Y. Zhou,Z. Xi, X. Wang, X. Fan, S. Pu, J. Zhu, R. Zheng, T. Gui,Q. Zhang, and X. Huang, “Loramoe: Revolutionizing mixtureof experts for maintaining world knowledge in languagemodel alignment,” ArXiv, vol. abs/2312.09979, 2023. [Online].Available: https://api.semanticscholar.org/CorpusID:266335873





[344] Y. Gui, X. Yan, P. Yin, H. Yang, and J. Cheng, “Spt:Fine-tuning transformer-based language models efficiently withsparsification,” ArXiv, vol. abs/2312.10365, 2023. [Online].Available: https://api.semanticscholar.org/CorpusID:266348310





[345] W. Niu, J. Guan, Y. Wang, G. Agrawal, and B. Ren, “Dnnfusion:accelerating deep neural networks execution with advanced op-erator fusion,” in Proceedings of the 42nd ACM SIGPLAN Interna-tional Conference on Programming Language Design and Implementa-tion, 2021, pp. 883–898.





[346] R. Y. Aminabadi, S. Rajbhandari, A. A. Awan, C. Li, D. Li,E. Zheng, O. Ruwase, S. Smith, M. Zhang, J. Rasley et al.,“Deepspeed-inference: enabling efficient inference of transformermodels at unprecedented scale,” in SC22: International Conferencefor High Performance Computing, Networking, Storage and Analysis.IEEE, 2022, pp. 1–15.





[347] J. Fang, Y. Yu, C. Zhao, and J. Zhou, “Turbotransformers: an effi-cient gpu serving system for transformer models,” in Proceedingsof the 26th ACM SIGPLAN Symposium on Principles and Practice ofParallel Programming, 2021, pp. 389–402.





[348] Y. Zhai, C. Jiang, L. Wang, X. Jia, S. Zhang, Z. Chen, X. Liu,and Y. Zhu, “Bytetransformer: A high-performance transformerboosted for variable-length inputs,” in 2023 IEEE InternationalParallel and Distributed Processing Symposium (IPDPS). IEEE,2023, pp. 344–355.





[349] Y. Sheng, L. Zheng, B. Yuan, Z. Li, M. Ryabinin, B. Chen, P. Liang,C. Re, I. Stoica, and C. Zhang, “Flexgen: High-throughput gen- ´erative inference of large language models with a single gpu,”in International Conference on Machine Learning. PMLR, 2023, pp.31 094–31 116.





[350] Y. Song, Z. Mi, H. Xie, and H. Chen, “Powerinfer: Fast largelanguage model serving with a consumer-grade gpu,” arXivpreprint arXiv:2312.12456, 2023.





[351] L. Zheng, Z. Li, H. Zhang, Y. Zhuang, Z. Chen, Y. Huang,Y. Wang, Y. Xu, D. Zhuo, E. P. Xing et al., “Alpa: Automating inter-and {Intra-Operator} parallelism for distributed deep learning,”in 16th USENIX Symposium on Operating Systems Design andImplementation (OSDI 22), 2022, pp. 559–578.





[352] M. Shoeybi, M. Patwary, R. Puri, P. LeGresley, J. Casper, andB. Catanzaro, “Megatron-lm: Training multi-billion parame-ter language models using model parallelism,” arXiv preprintarXiv:1909.08053, 2019.





[353] S. Li, H. Liu, Z. Bian, J. Fang, H. Huang, Y. Liu, B. Wang, andY. You, “Colossal-ai: A unified deep learning system for large-scale parallel training,” in Proceedings of the 52nd InternationalConference on Parallel Processing, 2023, pp. 766–775.





[354] M. Baines, S. Bhosale, V. Caggiano, N. Goyal, S. Goyal,M. Ott, B. Lefaudeux, V. Liptchinsky, M. Rabbat, S. Sheif-fer et al., “Fairscale: A general purpose modular pytorch li-brary for high performance and large scale training,” 2021,https://github.com/facebookresearch/fairscale.





[355] G. Lai, “Pax: A jax-based machine learning framework for largescale models.” https://github.com/google/paxml.





[356] J. Rasley, S. Rajbhandari, O. Ruwase, and Y. He, “Deepspeed:System optimizations enable training deep learning models withover 100 billion parameters,” in Proceedings of the 26th ACMSIGKDD International Conference on Knowledge Discovery & DataMining, 2020, pp. 3505–3506.





[357] T. M. M. Team, “composer,” https://github.com/mosaicml/composer/, 2021.





[358] A. Pham, C. Yang, S. Sheng, S. Zhao, S. Lee, B. Jiang, F. Dong,X. Guan, and F. Ming, “Openllm: Operating llms in production,”2023, https://github.com/bentoml/OpenLLM.





[359] T. A. B. Team, “Rayllm,” https://github.com/ray-project/ray-llm.





[360] M. team, “MLC-LLM,” 2023. [Online]. Available: https://github.com/mlc-ai/mlc-llm





[361] T. W. J. Team, “Saxml,” https://github.com/google/saxml.





[362] K. Yang, Z. Liu, and P. Cheng, “MOSEC: Model Servingmade Efficient in the Cloud,” 2021. [Online]. Available:https://github.com/mosecorg/mosec





[363] T. D. K. Team, “Llm foundry,”https://github.com/mosaicml/llm-foundry.





[364] TensorFlow, “Tensorflow xla,” https://www.tensorflow.org/xla.





[365] T. Chen, T. Moreau, Z. Jiang, L. Zheng, E. Yan, H. Shen,M. Cowan, L. Wang, Y. Hu, L. Ceze et al., “ $\{ \mathrm { \bar { T } V M } \}$ : An automated{End-to-End} optimizing compiler for deep learning,” in $_ { 1 3 t h }$USENIX Symposium on Operating Systems Design and Implementa-tion (OSDI 18), 2018, pp. 578–594.





[366] X. Jiang, H. Wang, Y. Chen, Z. Wu, L. Wang, B. Zou, Y. Yang,Z. Cui, Y. Cai, T. Yu et al., “Mnn: A universal and efficientinference engine,” Proceedings of Machine Learning and Systems,vol. 2, pp. 1–13, 2020.





[367] Pytorch, “Pytorch jit,” https://github.com/pytorch/torchdynamo.





[368] N. Houlsby, A. Giurgiu, S. Jastrzebski, B. Morrone, Q. De Larous-silhe, A. Gesmundo, M. Attariyan, and S. Gelly, “Parameter-efficient transfer learning for nlp,” in International Conference onMachine Learning. PMLR, 2019, pp. 2790–2799.





[369] A. Ruckl ¨ e, G. Geigle, M. Glockner, T. Beck, J. Pfeiffer, N. Reimers, ´and I. Gurevych, “Adapterdrop: On the efficiency of adapters intransformers,” arXiv preprint arXiv:2010.11918, 2020.





[370] J. Pfeiffer, A. Kamath, A. Ruckl ¨ e, K. Cho, and I. Gurevych, ´“Adapterfusion: Non-destructive task composition for transferlearning,” arXiv preprint arXiv:2005.00247, 2020.





[371] S. He, L. Ding, D. Dong, M. Zhang, and D. Tao, “Sparseadapter:An easy approach for improving the parameter-efficiency ofadapters,” arXiv preprint arXiv:2210.04284, 2022.





[372] X. L. Li and P. Liang, “Prefix-tuning: Optimizing continuousprompts for generation,” arXiv preprint arXiv:2101.00190, 2021.





[373] E. B. Zaken, S. Ravfogel, and Y. Goldberg, “Bitfit: Simpleparameter-efficient fine-tuning for transformer-based maskedlanguage-models,” arXiv preprint arXiv:2106.10199, 2021.





[374] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang,and W. Chen, “Lora: Low-rank adaptation of large languagemodels,” arXiv preprint arXiv:2106.09685, 2021.





[375] T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer,“Qlora: Efficient finetuning of quantized llms,” arXiv preprintarXiv:2305.14314, 2023.





[376] Y. Xu, L. Xie, X. Gu, X. Chen, H. Chang, H. Zhang, Z. Chen,X. Zhang, and Q. Tian, “Qa-lora: Quantization-aware low-rank adaptation of large language models,” arXiv preprintarXiv:2309.14717, 2023.





[377] M. Valipour, M. Rezagholizadeh, I. Kobyzev, and A. Ghodsi,“Dylora: Parameter efficient tuning of pre-trained models us-ing dynamic search-free low-rank adaptation,” arXiv preprintarXiv:2210.07558, 2022.





[378] Q. Zhang, M. Chen, A. Bukharin, P. He, Y. Cheng, W. Chen, andT. Zhao, “Adaptive budget allocation for parameter-efficient fine-tuning,” arXiv preprint arXiv:2303.10512, 2023.





[379] R. Karimi Mahabadi, J. Henderson, and S. Ruder, “Compacter:Efficient low-rank hypercomplex adapter layers,” Advances inNeural Information Processing Systems, vol. 34, pp. 1022–1035, 2021.





[380] V. Lialin, V. Deshpande, and A. Rumshisky, “Scaling downto scale up: A guide to parameter-efficient fine-tuning,” arXivpreprint arXiv:2303.15647, 2023.



# APPENDIX A

# PARAMETER-EFFICIENT FINETUNING (PEFT)

When training a large language model for specific tasks,it is often faster and more efficient to fine-tune a pre-trained model. There are two types of fine-tuning methods:full fine-tuning and parameter-efficient fine-tuning (PEFT).However, full fine-tuning can be expensive and may causecatastrophic forgetfulness problems. To address these issues,PEFT was developed. PEFT methods can be categorized intothree categories: additive, selective, and reparameterization.

# A.1 Additive methods

The main idea behind additive methods is to add additionalparameters to the model while keeping the original param-eters fixed and train the model by fine-tuning the additionalparameters [368], [369], [370], [371], [372].

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-11/0a775777-6cdd-4dbe-a304-295f64b9a1ad/4fc58b4e897ae72c3fb310c0b3a68d923ad74d9070b2ca800bcc855a849caa92.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-11/0a775777-6cdd-4dbe-a304-295f64b9a1ad/9aa5f233ac5775278780d4e39f3f5fe429bc2d925359364f991adf5692dcac45.jpg)



Fig. 10: Architecture of the adapter module and its integra-tion with the Transformer.


The Adapter [368] is the pioneering work of the additivemethods. As shown in Fig. 10, the Adapter fine-tunes themodel by inserting an adapter module between the feed-forward layer and skip-connection within the Transformerlayer. The Adapter module is a small fully-connected net-work consisting of an upper projection layer, a nonlinearlayer, a lower projection layer, and an additional skip-connection. Whenever a new downstream task arises, wecan mitigate the problem of full fine-tuning and catastrophicforgetting by adding an adapter module to the model toproduce an easily scalable downstream model. The adapteridea has been widely used and many adapter variants havebeen proposed.

There are two main strategies for integrating multi-taskknowledge: sequential fine-tuning and multi-task learning.However, both strategies have specific problems. Sequentialfine-tuning requires prior knowledge to determine the orderof tasks which can result in the model forgetting knowledgelearned from previous tasks. On the other hand, multi-tasklearning makes balancing data from various tasks challeng-ing as different tasks can interfere with each other. As a re-sult, both methods face challenges in effectively transferringknowledge. [370]proposes a variant called AdapterFusion,which effectively mitigates the above problems in multi-task training using a two-stage learning algorithm, achieveseffective knowledge sharing across multiple tasks, and out-performs fully fine-tuned models on a single target task.

Typically, fine-tuning a Transformer with the adapteris $6 0 \%$ faster than full fine-tuning in training but $4 { - } 6 \%$slower in inference. The AdapterDrop method proposed by[369] can efficiently and dynamically remove adapters withminimal impact on task performance. This dramaticallyimproves the model’s efficiency during backpropagation(training) and forward propagation (inference).

Existing methods increase the bottleneck dimension tomatch the performance of full fine-tuning as much as possi-ble because the number of trainable parameters determinesthe adapter’s capacity. However, these methods increase theoverall parameter count and FLOPs, violating the adapter’s

original intention. [371] combines the adapter with pruningto propose SparseAdapter, using a frustratedly easy settingcalled Large-Sparse, which effectively improves the capacityof the adapter under a given parameter budget and can alsobe effectively applied to other types of adapter methods,such as LoRA.

Another similar class of additive methods is SoftPrompts, of which the most representative work is Prefix-Tuning [372], which effectively avoids data cross-pollinationand dramatically reduces the number of parameters byadding a continuous, independent, learnable, task-specificprefix to the input. At the same time, it can easilyswitch tasks and realize processing examples from multipleusers/tasks in one batch.

Overall, while additive methods add additional param-eters to the model and inevitably bring about additionalinference delays, they significantly improve training speedand memory efficiency by reducing the gradient size andoptimizer state.

# A.2 Selective methods

Selective methods select a portion of the parameters on theoriginal model for fine-tuning and keep the rest frozen, suchas BitFit [373], which selectively fine-tunes only the model’sbias terms. It is more suitable for cases with less trainingdata because of the limited expression of the parameters ittunes.

These methods are relatively more straightforward toimplement. Still, the problem is that we can only deter-mine which part to tune empirically and experimentally,which can be inefficient and inaccurate.In addition, sincethese methods modify the original model, we also need toconsider the problem of model forgetting.

# A.3 Reparameterization methods

The reparameterization methods take advantage of low rankin the weight matrix of the pre-trained model. Instead offine-tuning the entire weight matrix, these methods con-struct smaller fine-tuned modules by reparameterization[374], [375], [376], [377], [378], [379].

The most representative work in this area is LoRA [374],which approximates the entire weight matrix by fine-tuningthe two-rank decomposition matrices instead of fine-tuningthem. In this case, the rank $r$ of the decomposition matricesis a hyperparameter; the larger $r$ is, the more parametersneed to be fine-tuned and the larger the capacity of thedecomposition matrices, and vice versa. Due to the low-rank nature of the pre-trained model’s weight matrix, it isgenerally sufficient for us to set r relatively small, bettermitigating the forgetting problem. In addition, when rea-soning, we can merge the decomposition matrix with theweight matrix to avoid introducing inference delays. Whenswitching tasks, we must subtract the original parts and addnew ones.

There are two problems with LoRA. The first is thatits size is fixed, which means that once we have set therank and trained on it, it cannot be modified anymore. Thesecond is that it is difficult to find the optimal rank unless weperform a search, which is expensive. [377] has developeda new algorithm called DyLoRA based on LoRA. DyLora

has developed a unique algorithm called DyLoRA basedon LoRA. DyLora has developed a new algorithm calledDyLoRA based on LoRA. DyLora has developed a newalgorithm called DyLoRA based on LoRA. It dynamicallysearches for the optimal rank through LoRA selects it, andperforms dynamic inference without incurring additionalcosts.

Another related work is AdaLoRA [378], which adap-tively assigns parameter budgets based on importancescores. In AdaLoRA, the incremental update of the weightmatrix is parameterized in the form of a singular valuedecomposition. The parameter budget is dynamically al-located among the total matrices by manipulating the dis-tinct values according to the new importance metrics. Thismethod effectively improves model performance and pa-rameter efficiency.

LoRA has another advantage: it can be naturally orthog-onalized to other model compression acceleration meth-ods. As mentioned earlier, QLoRA [375], which combinesLoRA with quantization, dramatically reduces the mem-ory footprint during training, realizing the advancementof fine-tuning 65B parametric models on a single 48GBmemory GPU, making it possible for more researchers toparticipate in the study of large models. However, QLoRAonly considers resources at training, not inference. AlthoughQLoRA quantizes the model during the training process,since the LoRA parameters for training are of FP16 type,when reasoning, the quantized model is fused with theLoRA parameters, the quantization will be destroyed andgo back to the unquantized form, and then if you want toreason efficiently, you have to perform another step of PTQ(Post-training quantization). PTQ will bring extra errors andaffect the performance of the model. Recently, QALoRA[376] solved the problem by introducing group-wise oper-ators, which realize one-click low-precision deployment ofmodels.

# A.4 Hybrid methods

Combining the previous methods is a natural idea to get bet-ter results. Compacter [379] combines the ideas of Adapterand reparameterization to balance the number of trainableparameters, task performance, and memory footprint. Theabove is a brief overview of representative work on PEFT.For more detailed information, see [380].

PEFT methods offer an effective way to train models, butit’s still difficult to deploy trained models on edge deviceslike cell phones. We can tweak the models before compres-sion, which may affect their performance. Therefore, it’simportant to explore new PEFT methods and find waysto combine them with other compression acceleration tech-niques to compress models while minimizing performanceloss in downstream tasks. This remains a valuable area ofresearch.