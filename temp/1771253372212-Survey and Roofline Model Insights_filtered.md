# LLM Inference Unveiled: Survey and Roofline Model Insights

<!-- IMAGE:1 -->



1Infinigence-AI , 2Illinois Institute of Technology, 3Carnegie Mellon University, 4Peking University, 5Tencent AI Lab,6Institute of Automation, CAS, 7University of Wisconsin, Madison, 8University of California, Berkeley.


# Abstract

The field of efficient Large Language Model (LLM) in-ference is rapidly evolving, presenting a unique blend of op-portunities and challenges. Although the field has expandedand is vibrant, there hasn’t been a concise framework thatanalyzes the various methods of LLM Inference to providea clear understanding of this domain. Our survey standsout from traditional literature reviews by not only summa-rizing the current state of research but also by introducing aframework based on Roofline model for systematic analysisof LLM inference techniques. This framework identifies thebottlenecks when deploying LLMs on hardware devices andprovides a clear understanding of practical problems, suchas why LLMs are memory-bound, how much memory andcomputation they need, and how to choose the right hard-ware. We systematically collate the latest advancementsin efficient LLM inference, covering crucial areas such asmodel compression (e.g., quantization), algorithm improve-ments (e.g., speculative decoding), and both system andhardware-level enhancements (e.g., operator fusion). Oursurvey stands out by analyzing these methods with Rooflinemodel, helping us understand their impact on memory ac-cess and computation. This distinctive approach not onlyshowcases the current research landscape but also deliversvaluable insights for practical implementation, positioningour work as an indispensable resource for researchers newto the field as well as for those seeking to deepen their un-derstanding of efficient LLM deployment. The analyze tool,LLM-Viewer, is open-sourced.

# 1. Introduction

Large Language Models (LLMs) have become a corner-stone of AI advancement in recent years, reshaping thelandscape of machine learning and natural language pro-

<!-- IMAGE:2 -->



Figure 1. Workflow of our designed LLM-Viewer. Input: theconfiguration details for the intended LLM deployment and spe-cific hardware device information. Upon receiving these inputs,the LLM-Viewer is designed to precisely analyze and identify thebottlenecks associated with deploying the given LLM on the spec-ified hardware device, facilitating targeted optimizations for effi-cient LLM inference.


cessing (NLP) [Zhao et al., 2023]. This trend can betraced to the success of revolutionary models like Chat-GPT [Brown et al., 2020, Ouyang et al., 2022], which pro-duce very human-like text through their exceptional un-derstanding and generation abilities. Following ChatGPT,other notable LLMs such as OPT [Zhang et al., 2022],BLOOM [Scao et al., 2022], and Llama [Touvron et al.,2023a,b] have emerged, further solidifying the consen-sus that larger models often lead to enhanced capabilities.Therefore, models with tens of billions of parameters arebecoming increasingly common. As a result of the vast sizeof these models, they present considerable inference chal-lenges, not only for devices with limited computational ca-pabilities, but also for the most advanced hardware. Be-cause of their complexity and scale, as well as their en-ergy and computational demands, these models are diffi-

<!-- IMAGE:3 -->



Figure 2. Mind-map of the Survey on Efficient LLM Inference. Our survey diverges from traditional surveys by focusing on the practicalaspects of LLM inference. Specifically, we identify and analyze the challenges associated with LLM inference. Subsequently, we introducea specially developed Roofline model to pinpoint the bottlenecks in LLM inference processes (Sec.2). The survey categorizes strategiesfor improving LLM inference efficiency into four main areas: Parameter Reduction (Sec.3), Fast Decoding Algorithm Design (Sec.4),System-Level Optimization (Sec.5), and Hardware-Level Optimization (Sec.6), providing a comprehensive framework for addressing thecomplexities of efficient LLM deployment.


cult to deploy in real-world situations. Additionally, theresource-intensive nature of these models raises concernsabout energy consumption, scalability, and accessibility.The situation is particularly challenging for smaller orga-nizations and communities with fewer computing resourcesthan large corporations. Therefore, these challenges empha-size the need for innovative solutions to make LLM infer-ence more universally accessible and sustainable.

Numerous methods have been developed to address thechallenges of deploying LLM. The field of efficient LLMinference has grown exponentially in the last two years,presenting both opportunities and challenges. While theburgeoning volume of research demonstrates the field’s vi-brancy, it can inadvertently mask key trends and slow ad-vancements. A critical gap in existing literature is theabsence of a systematic and practical framework for uni-fied analysis and comprehensive solution development. Tobridge this gap, our work offers a comprehensive overview

of the current state of research in efficient LLM inference,with a unique focus on its practice-driven characteristics.Diverging from traditional literature reviews, our work notonly discusses existing research but also introduces a specif-ically developed Roofline model. This model is designedto analyze bottlenecks in LLM deployments, a crucial stepwe believe is vital for practical application and optimiza-tion as shown in Figure 1. Our work is the first, to ourknowledge, that provides such a tool for analyzing the in-tricacies of inferring LLMs on hardware devices, system-atically collating the latest advancements in efficient LLMinference. We delve deep into deployment challenges, par-ticularly emphasizing inference efficiency. Our discussionspans various areas, including model compression, decod-ing algorithm refinement, system-level and hardware-levelenhancements, as illustrated in Figure 2. While there areconcurrently related surveys in this domain, such as [Zhuet al., 2023] on LLM compression and [Miao et al., 2023a],

[Ding et al., 2023] and [Wang et al., 2024a] on holistic LLMserving, our work stands out by incorporating a Rooflinemodel analysis.

In this paper, we first discuss the foundations ofLLMs and develop a tool named LLM-Viewer, which usesRoofline model to analyze the bottleneck of deployingLLMs (Sec.2). LLM-Viewer can be used to analyze thedeployment of any LLM architecture on various hardwareplatform, as shown in Figure 1. For the literature re-view, this survey categorizes strategies for improving LLMinference efficiency into four main areas: Model Com-pression (Sec.3), Algorithmic Methods for Fast Decoding(Sec.4), Compiler/System-Level Optimization (Sec.5), andHardware-Level Optimization (Sec.6).

# 2. Delve into LLM Inference and Deployment

# 2.1. LLM Inference

Nowadays, the prevailing architecture adopted by mostlarge language models (LLMs) is the Transformer decoderarchitecture. Here we will provide a concise overview of itsfundamental structure, with the option to refer to this sur-vey Zhao et al. [2023] for a more in-depth understanding.This structure comprises an embedding layer, a series of se-quential Transformer layers, and a prediction head. Figure 3demonstrated the architecture.

The embedding layer transform input tokens into the hid-den states. The hidden states are sent to the Transformerlayers. Each Transformer layer consists of two components.Firstly, there is a masked multi-head attention module, de-noted as MHA. Following MHA is a multi-layer percep-tron submodule, labeled as MLP. The output from the lastTransformer layer is then sent to the prediction head, whichis responsible for predicting the next token after the inputtokens.

Inference represents the process opposite to the train-ing process. During training, a model learns from a vastdataset to capture the intricacies of language and context.The weights in model are updated. In contrast, during in-ference, a user inputs a prompt, and the LLM engages in aprocess of generating responses. This process involves themodel utilizing its fixed pre-trained weights to comprehendthe input text and produce text as output. The inference pro-cess of Large Language Models (LLMs) is divided into twostages: the Prefill Stage and the Decode Stage.

The Prefill Stage serves as the initial step in LLM infer-ence. In this stage, the model takes a prompt sequence asinput and engages in the generation of a key-value cache(KV cache) for each Transformer layer within the LLM.The KV cache plays a crucial role in storing and organizinginformation that the model deems relevant for subsequenttoken generation. Each Transformer layer is equipped withits own unique KV cache, and this prefilling process estab-

lishes the foundation for the subsequent decoding stage.

In the Prefill Stage, the Multi-Head Attention (MHA)creats key-value (KV) pairs that will be stored in the KVcache. Let’s denote the input to a Transformer layer as$\mathbf { X } _ { \mathrm { p r e } } \in \mathbb { R } ^ { n \times d }$ , where $d$ is the hidden size and $n$ is the lengthof prompt token sequence. The layers in the MHA haveweights represented by $\mathbf { W } _ { q }$ , $\mathbf { W } _ { k }$ , $\mathbf { W } _ { v }$ , and $\mathbf { W } _ { o }$ . The query,key and value are computed through the following process:

$$
\text {Q u e r y :} \quad \mathbf {Q} _ {\mathrm {p r e}} = \mathbf {X} _ {\mathrm {p r e}} \cdot \mathbf {W} _ {q}
$$

$$
\text {K e y :} \quad \mathbf {K} _ {\text {p r e}} = \mathbf {X} _ {\text {p r e}} \cdot \mathbf {W} _ {k}
$$

$$
\text {V a l u e :} \quad \mathbf {V} _ {\mathrm {p r e}} = \mathbf {X} _ {\mathrm {p r e}} \cdot \mathbf {W} _ {v}
$$

The generated $\bf { K } _ { \mathrm { p r e } }$ and $\mathbf { V } _ { \mathrm { p r e } }$ are stored in the KV cache.The other computation in MHA can be formulated as 1:

$$
\mathbf {O} _ {\text {p r e}} = \operatorname {s o f t m a x} \left(\frac {\mathbf {Q} _ {\text {p r e}} \cdot \mathbf {K} _ {\text {p r e}} ^ {T}}{\sqrt {d}}\right) \cdot \mathbf {V} _ {\text {p r e}} \cdot \mathbf {W} _ {o} + \mathbf {X} _ {\text {p r e}},
$$

where the output of MHA $\mathbf { O } _ { \mathrm { p r e } } \in \mathbb { R } ^ { n \times d }$ is sent to the MLP.The output of the MLP serves as the input for the nextTransformer layer.

The Decode Stage represents the core of the LLM infer-ence process. In the Decode Stage, the model uses the KVcaches prepared earlier and might add new information tothem. The goal here is to generate tokens, which are essen-tially words or parts of words. This happens step by step.The creation of each new token is influenced by the tokensthat were generated before it, like building a sentence wordby word.

In the Decode Stage, the MHA loads the previouslystored KV cache $\mathbf { K } _ { \mathrm { c a c h e } }$ and $\mathbf { V } _ { \mathrm { c a c h e } }$ . The input is $\mathbf { X } _ { \mathrm { d e c } } \in$$\mathbb { R } ^ { 1 \times d }$ . New key and value pairs are computed and concate-nated to the existing cache:

$$
\text {Q u e r y :} \quad \mathbf {Q} _ {\mathrm {d e c}} = \mathbf {X} _ {\mathrm {d e c}} \cdot \mathbf {W} _ {q}
$$

$$
\text {K e y :} \quad \mathbf {K} _ {\mathrm {c a t}} = \left[ \mathbf {K} _ {\mathrm {c a c h e}}, \mathbf {X} _ {\mathrm {d e c}} \cdot \mathbf {W} _ {k} \right]
$$

$$
\text {V a l u e :} \quad \mathbf {V} _ {\mathrm {c a t}} = \left[ \mathbf {V} _ {\mathrm {c a c h e}}, \mathbf {X} _ {\mathrm {d e c}} \cdot \mathbf {W} _ {v} \right]
$$

These newly computed $\mathbf { X } _ { \mathrm { d e c } } \cdot \mathbf { W } _ { k }$ and $\mathbf { X } _ { \mathrm { d e c } } \cdot \mathbf { W } _ { v }$ are thenappended to the KV cache. The other computation in MHAis carried out as follows:

$$
\mathbf {O} _ {\mathrm {d e c}} = \operatorname {s o f t m a x} \left(\frac {\mathbf {Q} _ {\mathrm {d e c}} \cdot \mathbf {K} _ {\mathrm {c a t}} ^ {T}}{\sqrt {d}}\right) \cdot \mathbf {V} _ {\mathrm {c a t}} \cdot \mathbf {W} _ {o} + \mathbf {X} _ {\mathrm {d e c}}
$$

where the output of MHA $\mathbf { O } _ { \mathrm { d e c } } \in \mathbb { R } ^ { 1 \times d }$ is sent to the MLP.The output of the last Transformer layer is sent to the finalprediction layer to predict the next token.

<!-- IMAGE:4 -->



Figure 3. Demonstration of the architecture of LLMs.


<!-- IMAGE:5 -->



Figure 4. Execution of an operation on hardware.


# 2.2. Roofline Model

Assessing the efficiency at which LLMs deploy onto spe-cific hardware involves a comprehensive consideration ofboth hardware and model characteristics. To conduct thisevaluation, we employ the Roofline model. The Rooflinemodel serves as an effective theoretical framework to assessthe potential performance of deploying a model on particu-lar hardware.

As shown in Figure 4, the execution of a neural networklayer on hardware devices entails the transfer of data frommemory (DDR or HBM) to on-chip buffers, followed bycomputations performed by on-chip processing units, ulti-mately outputting results back to memory. Therefore, eval-uating performance requires simultaneous consideration ofmemory access and processing unit capabilities. If a layerinvolves extensive computations but minimal memory ac-cess, it is termed a computation bottleneck. This scenarioleads to idle on the memory access. On the contrary, whena layer requires substantial memory access with fewer com-putational demands, it is referred to as a memory bottle-neck. In this case, computational units remain underuti-lized. We can clearly distinguish between these two sce-narios according to the Roofline model and provide perfor-

<!-- IMAGE:6 -->



Figure 5. Demonstration of the Roofline model of Nvidia A6000GPU. The computation is in FP16.


mance upper bounds for different situations.

There are two steps to using the Roofline model:

1. Plot the Roofline: Determine the peak computationalperformance (operations per second, OPS) and peak mem-ory bandwidth (bytes per second) specific to the target hard-ware device.2 Then create a graph with performance (OPS)on the y-axis and arithmetic intensity (OPs/byte) on the x-axis: Draw a horizontal line equal to the peak computationalperformance. This line represents the maximum achievableperformance by the hardware device. And draw a diagonalline from the origin with a slope equal to the peak mem-ory bandwidth. This line represents the maximum mem-ory bandwidth available on the system, known as the mem-ory Roofline. Figure 5 demonstrates the Roofline model ofNvidia A6000 GPU.


Table 1. Analysis for layers in Llama-2-7b using the Rooflinemodel of Nvidia A6000 GPU. In this example, the sequence lengthis 2048 and the batch size is 1.


<!-- TABLE:1 -->

2. Analyze performance for layers: Evaluate the per-formance of each layer in the model by quantifying boththe number of operations (OPs) and the volume of data ac-cessed from memory (bytes). Calculate the arithmetic in-tensity (OPs/byte) of each layer by dividing the requiredoperations by the amount of data transferred. According tothe graph created in the first step, the theoretical max per-formance for each layer is determined by the position onthe graph corresponding to the x-axis value of arithmeticintensity. It allows us to ascertain whether the system ismemory-bound or compute-bound at this point, guiding thedetermination of the subsequent optimization strategy.

There are two scenarios where resources are not fullyutilized: When the model’s computational intensity is be-low the turning point, residing in the red zone, it impliesthat the computational workload required per memory ac-cess is low. Even saturating the peak bandwidth does notfully utilize all computational resources. In such cases, thelayer is constrained by memory access (memory-bound),and some computational units may remain idle. If the layeris memory-bound, consider optimization techniques such asquantization, kernel fusion and increasing batch size to al-leviate the memory footprint. Conversely, if the model’scomputational intensity is above the turning point, situated

in the green zone, it suggests that the model requires onlya small amount of memory access to consume a significantamount of computational capability. It implies that the layeris constrained by computation (compute-bound), with somememory units potentially remaining idle. In this case, weshould investigate strategies such as enabling low-bit com-putation to enhance computational efficiency. Detailed ex-planations of these methods will be provided in the subse-quent sections.

As an example, Table 1 presents the analysis of lay-ers in Llama-2-7b using the Roofline model on the NvidiaA6000 GPU. From the table, we observe that during the pre-fill stage, the majority of computations are compute-bound,leading to high performance. Conversely, in the decodestage, all computations are memory-bound, resulting in per-formance significantly below the computational capacity ofthe GPU’s computation units. During the user’s interac-tion with large models, the prefill stage executes only once,while the decode stage is repeatedly performed to generate acontinuous output. Therefore, optimizing for the memory-bound characteristics of the decode stage becomes crucialfor enhancing the inference performance of large models.

# 2.3. LLM-Viewer

There are multiple Transformer layers in LLMs, each con-taining various operations. Moreover, different LLMs havedifferent sets of operations. Additionally, we need to trackinformation like memory footprint to calculate the peakmemory usage and total inference time. Hence, analyzingLLMs involves examining network-wide concerns. In thissection, we propose a powerful tool, LLM-Viewer 3, to ex-ecute the network-wise analysis. It enables the analysis ofLLM performance and efficiency on various hardware plat-forms, offering valuable insights into LLM inference andperformance optimization.

The workflow of LLM-Viewer is depicted in Figure 1.It consists of the following steps: (1) Input the LLM andgather essential information about each layer, such as thecomputation count, input and output tensor shapes, and datadependencies. (2) Provide input for the hardware and gen-erate a Roofline model that takes into account the computa-tion capacity and memory bandwidth of the hardware. (3)Configure the inference settings, including the batch size,prompt token length, and generation token length. (4) Con-figure the optimization settings, such as the quantizationbitwidth, utilization of FlashAttention, decoding methods,and other system optimization techniques. (5) The LLM-Viewer Analyzer utilizes the Roofline model and layer in-formation to analyze the performance of each layer. It alsotracks the memory usage of each layer and calculates thepeak memory consumption based on data dependencies. Byaggregating the results of all layers, the overall network

performance of LLM can be obtained. (6) Generate a re-port that provides information such as the maximum per-formance and performance bottlenecks of each layer andthe network, as well as the memory footprint. Analyz-ing curves, such as batch size-performance and sequencelength-performance curves, can be plotted from the report tounderstand how different settings impact performance. (7)LLM-Viewer offers a web viewer that allows convenient vi-sualization of the network architecture and analysis results.This tool facilitates easy configuration adjustment and pro-vides access to various data for each layer.

# 3. Model Compression

The formidable size and computational demands of LargeLanguage Models (LLMs) present significant challengesfor practical deployment, especially in resource-constrainedenvironments. To alleviate these limitations, the moststraightforward solution is to compress the LLMs. In thissection, we review the concept of neural network compres-sion for LLMs. This exploration encompasses a thoroughexamination of well-established techniques, including butnot limited to quantization, pruning, knowledge distillation,and low-rank factorization. In each subsection, we will uti-lize LLM-Viewer to analyze the impact of network com-pression on LLM inference. Based on our analysis, we willprovide optimization recommendations.

# 3.1. Quantization

In the realm of LLM compression, quantization has be-come a pivotal technique for mitigating the substantial stor-age and computational overhead associated with these mod-els. Essentially, quantization involves transforming thefloating-point values in original LLMs into integers or otherdiscrete forms, a process that considerably reduces bothstorage requirements and computational complexity [Gho-lami et al., 2022]. While some degree of precision lossis inherent in this process, carefully designed quantiza-tion techniques can achieve significant model compressionwith minimal impact on accuracy. Quantization in the con-text of LLMs can be primarily categorized into two di-rections: Quantization for Compressing Pre-trained LLMsand Quantization for Parameter-Efficient Fine-Tuning (Q-PEFT). The first category encompasses approaches that ap-ply quantization to LLMs for using the quantized LLMsas pre-trained models. This category can be further di-vided into two subcategories: Quantization-Aware Train-ing (QAT) and Post-Training Quantization (PTQ). QAT in-tegrates quantization into the model’s training process orduring the fine-tuning/re-training of a pre-trained LLM, al-lowing the model to adapt to the quantization from the on-set. In contrast, PTQ applies quantization to a model afterit has completed its training phase, offering a more straight-forward approach to model compression without the need

<!-- IMAGE:7 -->



Figure 6. Demonstration of the Roofline model of Nvidia A6000GPU for different computation data types.


for retraining. These distinct methodologies highlight theversatility of quantization techniques in addressing the spe-cific needs and constraints of LLM deployment.

# 3.1.1 A Use Case of LLM-Viewer:Roofline Analysis for Quantization

Here we provide an example of how to use our LLM-Viewer(Section 2.3) to analyze the bottlenecks of LLM deploy-ments. In LLMs, tensors consist of weights and activa-tions, with activations including temporary activations andKV cache. (1) LLM weights must be stored in memory.For example, Llama-13b [Touvron et al., 2023a], which has13 billion weights, occupies approximately 26GB of mem-ory in FP16 format. (2) temporary activations are generatedduring inference. For example, the inputs of each trans-former layer are kept in memory until the residual additionis executed. (3) for auto-regressive LLMs, caching key andvalue activations (KV cache) into memory is necessary forsubsequent token generation. We utilize LLM-Viewer toanalyze the effects of quantization on these tensors fromthree perspectives: computation, memory consumption, andmemory access.

Computation: The latest computing devices, such asNVIDIA GPUs, generally support FP32, FP16, and INT8data types for computation. Hardware devices gener-ally perform better when processing data with smaller bitwidths. NVIDIA’s A6000 GPU, for example, is capable ofperforming twice as fast as FP16 with 155 TOP/s and 310TOP/s, respectively. In the Roofline model, when enablingquantization for faster computation, the roofline heightincreases, indicating improved performance for compute-bound layers. As shown in Figure 6, the max performanceimproved when using INT8 computation. However, to uti-lize the computational power of INT8, all input operandsmust be in INT8 format. Consequently, if only the weightsare quantized to INT8 while the activations remain in FP16

<!-- IMAGE:8 -->


<!-- IMAGE:9 -->



Figure 7. Relative memory consumption for different quantizationsettings for Llama-2-13b. Tmp Act means temporary activations.


format, the INT8 computational power cannot be utilized.Instead, the INT8 weights would need to be converted toFP16 for multiplication with FP16 activations. Further-more, when tensors are quantized to a bitwidth that isnot supported by the hardware, they need to be convertedto higher bit widths for computation. For example, theNVIDIA H100 GPU does not support INT4 computation.Consequently, if the weight or activation is quantized toINT4, it would require conversion to a higher bit width,such as INT8 or FP16, for computation.

Memory Consumption: The memory consumption re-duction resulting from quantizing different tensors varies, asshown in Figure 7 4. Notably, the memory usage of tempo-rary activations is relatively low, especially during the de-code stage. This can be attributed to their short lifespan,allowing their memory to be released once their purposeis fulfilled. On the other hand, the memory allocated forthe KV cache behaves differently. It cannot be freed untilthe entire process of generating a complete answer is fin-ished, which entails multiple inference passes through thenetwork. Additionally, the memory consumption of the KVcache increases as the batch sizes grow larger and the in-put sequences become longer. This is because the modelneeds to store a greater number of key-value (KV) pairs tofacilitate its operations.

<!-- IMAGE:10 -->



Figure 8. Inference time of decoding stage for different quantiza-tion settings on Llama-2-13b. (Sequence length=1024)


Memory Access: Quantizing tensors in LLM can signif-icantly reduce memory access, resulting in fewer data bytesto be moved for the same amount of computation. Thisincrease in arithmetic intensity contributes to the Rooflinemodel, leading to three scenarios: (1) After quantization,the arithmetic intensity remains within the memory-boundrange. With the improvement in arithmetic intensity, theaverage data access per computation is reduced, alleviatingthe pressure on data memory access. Consequently, the the-oretical performance is enhanced. This can greatly boostthe performance during the memory-bound decode stage.(2) The arithmetic intensity transitions from being memory-bound to compute-bound. This shift also reduces the pres-sure on data memory access, resulting in improved theoret-ical performance. (3) Both before and after quantization,the arithmetic intensity remains within the compute-boundrange. In this case, there is no performance improvement.For example, this scenario can occur during the compute-bound prefill stage or when the batch size is large in thedecode stage.

As depicted in Figure 8, when the batch size is small, thelayers in the network are memory-bound both before and af-ter quantization. Therefore, quantization can enhance per-formance and reduce the network’s inference time. How-ever, when the batch size is large, compressing the net-work’s weights from 4 bits to 2 bits or 1 bit does not leadto a decrease in the inference time. This is because, at thispoint, the network is already compute-bound, and quantiz-ing the weights becomes ineffective. Similar to the previ-ous scenario, the behavior of the system can exhibit satu-ration effects in prefill stage. As shown in Figure 9, whenthe sequence length is relatively small, the prefill stage ismemory-bound. In this case, applying quantization can en-hance the performance by reducing the memory access re-quirements of the network. However, as the sequence lengthincreases, the prefill stage becomes more compute-bound.Consequently, quantizing the weights may not yield signifi-cant improvements in performance when the network is al-ready compute-bound during the prefill stage with large se-quence lengths.

<!-- IMAGE:11 -->



Figure 9. Inference time of prefill stage for different quantizationsettings on Llama-2-13b. (Batch size ${ \mathrm { : = } } 1$ )


# 3.1.2 Quantization for Compressing Pre-trainedLLMs

In Quantization-Aware Training (QAT) [Choi et al., 2018,Courbariaux et al., 2015, Dong et al., 2019], the quanti-zation process is seamlessly integrated into the training ofLarge Language Models (LLMs), enabling them to adapt tolow-precision representations and thus mitigating precisionloss. LLM-QAT [Liu et al., 2023b] innovatively addressesthe challenge of training data acquisition for LLMs throughdata-free distillation, which leverages outputs from a pre-trained model to obviate the need for extensive data collec-tion. Furthermore, LLM-QAT expands quantization beyondweights and activations to include key value (KV) caches,enhancing throughput and supporting longer sequence de-pendencies. Its successful distillation of large Llama mod-els to 4-bit quantized weights and KV caches underscoresthe potential for accurately quantized 4-bit LLMs.

To attain lower-bit quantization, such as below 2-bit,Kim et al. [2023b] introduce Token-Scaled Logit Distil-lation (TSLD) for ternary QAT in LLMs. This methodemploys an adaptive knowledge distillation technique thatmodifies Logit Knowledge Distillation based on token con-fidence, providing tailored guidance during LLM QAT. Fur-thermore, Shang et al. [2024] focus on salient weights withtheir concept of partially binarized matrices in PB-LLM. Bypreserving these crucial weights in higher bits, PB-LLM ef-fectively maintains the reasoning capacity of heavily quan-tized LLMs. Additionally, PB-LLM explores minimizingquantization error by determining the optimal scaling fac-tors for binarized LLMs, a vital step in preserving the effec-tiveness of models under aggressive quantization.

# Post-Training Quantization (PTQ)

Post-Training Quantization (PTQ) represents a crucialtechnique in optimizing Large Language Models (LLMs),entailing the quantization of model parameters post theLLM’s training phase. The primary goal of PTQ is to re-duce both the storage requirements and computational com-plexity of the LLM, without necessitating alterations to themodel’s architecture or embarking on a retraining process.

This approach stands out for its simplicity and efficiency,particularly in achieving significant model compression. Inthe context of LLMs, which typically contain billions ofparameters, Quantization-Aware Training (QAT) often be-comes impractical due to excessive training costs. Hence,PTQ emerges as a more viable solution for these large-scalemodels. However, it’s crucial to acknowledge that PTQ canlead to a certain degree of precision loss as a consequence ofthe quantization process. Despite this, PTQ serves as an ef-fective method to enhance the efficiency of LLMs, offeringa straightforward solution that avoids major modificationsor extensive additional training.

In PTQ, various approaches focus on weight-onlyquantization to enhance efficiency. For instance, LUT-GEMM [Park et al., 2023] optimizes matrix multiplica-tions in LLMs using weight-only quantization and the BCQformat, thereby reducing latency and improving computa-tional efficiency. LLM.int8() [Dettmers et al., 2022] em-ploys 8-bit quantization, which halves GPU memory usageduring inference and maintains precision through vector-wise quantization and mixed-precision decomposition. Thismethod enables efficient inference in models up to 175 bil-lion parameters. ZeroQuant [Yao et al., 2022] combines ahardware-friendly quantization scheme with layer-by-layerknowledge distillation, optimizing both weights and acti-vations to INT8 with minimal accuracy loss. Addressinghigher compression targets, GPTQ [Frantar et al., 2022] in-troduces a layer-wise quantization technique based on ap-proximate second-order information, achieving a reductionto 3-4 bits per weight with minimal accuracy loss. Addition-ally, the study by Dettmers and Zettlemoyer [2023] exploresthe balance between model size and bit precision, partic-ularly for zero-shot performance, finding that 4-bit preci-sion generally offers the optimal balance. Innovations likeAWQ [Kim et al., 2023c, Lin et al., 2023] highlight thatprotecting a small percentage of salient weights can signif-icantly reduce quantization error. AWQ uses an activation-aware approach, focusing on weight channels with largeractivation magnitudes, and incorporates per-channel scalingfor optimal quantization. OWQ [Lee et al., 2023] analyzeshow activation outliers amplify quantization error, introduc-ing a mixed-precision scheme to assign higher precision toweights affected by these outliers. SpQR [Dettmers et al.,2023b] takes a unique approach by isolating outlier weightsfor storage in higher precision, while compressing the re-mainder to 3-4 bits. This technique allows for more efficientcompression while maintaining near-lossless performance.QuantEase [Behdin et al., 2023] suggests using a coordinatedescent approach to optimize all of the weights in network,improving the efficiency of quantization.

To achieve even lower-bit quantization (e.g., below 2-bit), QuIP [Chee et al., 2023] introduces an innovative ap-proach that accounts for the even distribution of weight

<!-- IMAGE:12 -->



Figure 10. Timeline of Quantization for LLM methods from 2022 to 2024. The red-highlighted methods represent they belonging to Quan-tization for Parameter Efficient Fine-Tuning (Q-PEFT), the green-highlighted methods represent they belonging to QAT-related methods,and others are PTQ-based methods.


magnitudes and the significance of accurately rounding di-rections unaligned with coordinate axes. QuIP comprisesan adaptive rounding procedure that minimizes a quadraticproxy objective, essential for optimizing the quantizationprocess. Additionally, it employs efficient pre- and post-processing techniques that ensure weight and Hessian inco-herence through multiplication by random orthogonal ma-trices, crucial for maintaining quantization effectiveness.Further advancing PTQ methods, Li et al. [2023a] are in-spired by the observation that aligning the quantized ac-tivation distribution with its floating-point counterpart canrestore accuracy in LLMs. Their proposed ’Norm Tweak-ing’ strategy involves a meticulous calibration data gener-ation process and a channel-wise distance constraint. Thisapproach updates the weights of normalization layers, lead-ing to enhanced generalization capabilities. [Shang et al.,2024] propose partial-binarized LLM (PB-LLM) by intro-ducing binarization [Hubara et al., 2016] into LLM quanti-zation to push weight quantization under 2 bits. FollowingPB-LLM, BiLLM [Huang et al., 2024] pushes weight quan-tization to almost 1 bit.

Apart from efforts that focus solely on weight quantiza-tion in LLMs, numerous PTQ approaches focus on weightsand activations quantization. SmoothQuant [Xiao et al.,2023a] addresses the challenge of quantizing activations,which can be complex due to the presence of outliers.It introduces a per-channel scaling transformation that ef-fectively smooths out activation magnitudes, rendering themodel more receptive to quantization. Recognizing the in-tricacies of quantizing activations in LLMs, RPTQ [Yuanet al., 2023c] highlights the uneven ranges across channelsand the prevalence of outliers. RPTQ’s innovative approachinvolves clustering channels for quantization, thereby re-ducing discrepancies in channel ranges. This methodsmartly integrates channel reordering into layer normal-ization and linear layer weights to minimize overhead.OliVe [Guo et al., 2023a] adopts an outlier-victim pair

(OVP) quantization strategy, focusing on local handling ofoutliers with low hardware overhead and significant perfor-mance benefits. This approach stems from the understand-ing that outliers are crucial, while adjacent normal valuesare less so. Building on this, Outlier Suppression+ extendsthe concept by addressing asymmetrically distributed harm-ful outliers in specific channels. It introduces channel-wiseshifting and scaling operations to balance the outlier dis-tribution and reduce the impact of problematic channels,considering both the nature of the outliers and the subse-quent quantization errors. ZeroQuant-FP [Wu et al., 2023d]delves into floating-point (FP) quantization, specifically ex-ploring FP8 and FP4 formats. This study finds that FP8activation quantization in LLMs outperforms the traditionalINT8 format, while FP4 weight quantization shows compa-rable efficacy to INT4. ZeroQuant-FP addresses the diver-gence between weights and activations by standardizing allscaling factors as powers of 2 and restricting them within asingle compute group, ensuring consistency and efficiencyin the quantization process. Li et al. [2023c] propose FPTQ,in which they employ a layerwise strategy to cope with dif-ferent levels of quantization difficulty. Particularly, they de-vises an offline logarithmic activation equalization to ren-der a quantization-friendly distribution for previously in-tractable layers.

Since the end of 2023, the length of tokens has beensignificantly increasing, causing the KV cache to consumemore memory. For instance, Google Gemini 1.5 [Sun-dar Pichai, 2024] can handle up to 1 million tokens inproduction, and LLMs processing books, large images orvideos will require tens of thousands of tokens. As a result,the optimization of KV Cache Quantization has becomeincreasingly important. Several recent papers in 2024 havefocused on improving KV cache quantization. For exam-ple, Hooper et al. [2024] propose a solution for achieving10 million context length LLM inference with KV CacheQuantization. KIVI [Liu et al., 2024b] pushes the quan-

<!-- IMAGE:13 -->



Figure 11. Memory Consumption of decode stage for differentquantization settings on Llama-2-13b. (Batch siz ${ \boldsymbol { \mathbf { \rho } } } { \boldsymbol { \mathbf { \rho } } } { \boldsymbol { \mathbf { \rho } } } { \boldsymbol { \mathbf { \rho } } } { \boldsymbol { \mathbf { \rho } } } { \boldsymbol { \mathbf { \rho } } } { \boldsymbol { \mathbf { \rho } } } { \boldsymbol { \mathbf { \rho } } } { \boldsymbol { \mathbf { \rho } } }$ ).


tization of KV cache to 2-bit. Yue et al. [2024] proposesWKVQuant as to jointly optimize the quantization of boththe weights and the KV cache in LLMs, making W4KV4have the same performance as W4. As shown in Figure 11,we use LLM-Viewer to analyze the memory reduction ofKV cache quantization. We can observe that when the se-quence length is larger than 50k, the KV cache takes most ofthe memory and its quantization can significantly decreasethe memory consumption.

# 3.1.3 Quantization for Parameter Efficient Fine-Tuning (Q-PEFT)

Parameter Efficient Fine-Tuning (PEFT) is an importanttopic for LLMs. One of the most popular approaches is low-rank adaptation (LoRA) [Hu et al., 2021, Valipour et al.,2022], where the key insight is to decompose the adapterweights into the multiplication of two low-rank (and thusparameter-efficient) matrices. LoRA has claimed compara-ble performance to full fine-tuning while using much fewerlearnable parameters. Please refer to the review paper [Huet al., 2023] for more details about this adaptor.

In addition to the well-defined quantization paradigms,a novel paradigm in LLM efficiency is emerging: Quanti-zation for Parameter-Efficient Fine-Tuning (Q-PEFT). Thisapproach integrates quantization into the fine-tuning pro-cess of LLMs, offering a unique and efficient method, par-ticularly relevant in the era of large models. Pioneeringworks in this paradigm, such as PEQA [Kim et al., 2023a],DFT [Li et al., 2023e], and QLORA [Dettmers et al., 2023a]demonstrate the feasibility and effectiveness of this ap-proach. PEQA employs a dual-stage process where the firststage involves quantizing the parameter matrix of each fullyconnected layer into a matrix of low-bit integers coupledwith a scalar vector. The second stage focuses on fine-tuning the scalar vector for specific downstream tasks, al-lowing for more efficient task-specific adjustments. DFTadopts the efficient Lion optimizer, which only keeps track

of the momentum and has consistent update magnitudes foreach parameter, an inherent advantage for robust quantiza-tion; and (ii) we quantize all model states and store themas integer values, and present a gradient flow and parame-ter update scheme for the quantized weights. On the otherhand, QLORA introduces novel concepts such as a new datatype, double quantization, and paged optimizers. These in-novations aim to conserve memory efficiently while main-taining LLM fine-tuning performance. Notably, QLORAfacilitates large model fine-tuning on a single GPU, achiev-ing state-of-the-art results on the Vicuna benchmark, a tes-tament to its effectiveness in balancing memory efficiencyand model performance.

However, a limitation of QLoRA is its restriction to atmost 4-bit quantization during fine-tuning; lower-bit quanti-zation, such as 2-bit, can significantly deteriorate the perfor-mance. Addressing this challenge, several studies have ven-tured into the realm of Q-PEFT to enable lower-bit quan-tization. LQ-LoRA [Guo et al., 2023b] introduces an it-erative algorithm that decomposes each pretrained matrixinto a high-precision, low-rank component and a memory-efficient quantized component. During fine-tuning, onlythe low-rank component is updated, keeping the quantizedcomponent fixed. This method presents an integer linearprogramming approach for the quantization component, al-lowing dynamic configuration of quantization parameterslike bit-width and block size within a given memory bud-get. Another notable approach, Loft-Q [Li et al., 2023d],simultaneously quantizes an LLM and establishes a suitablelow-rank initialization for LoRA fine-tuning. This strategyeffectively bridges the gap between the quantized and full-precision models, significantly enhancing generalization indownstream tasks. QA-LoRA [Xu et al., 2023c] leveragesthe benefits of quantizing the LLM’s weights into low-bitintegers, facilitating an efficient fine-tuning stage. Addi-tionally, it produces a lightweight, fine-tuned model, cir-cumventing the accuracy loss often associated with PTQ.

# 3.1.4 Discussion on LLM Quantiztaion

Figure 10 presents a timeline of LLM quantization tech-niques, highlighting the evolution from Post-TrainingQuantization (PTQ) as the initial mainstream approachto the rising prominence of Quantization-Aware Train-ing (QAT) and Quantization for Parameter-Efficient Fine-Tuning (Q-PEFT). This shift underscores the community’sadaptation in response to the performance bottlenecks en-countered with PTQ, marking QAT and Q-PEFT as the bur-geoning areas of focus in the quest for efficient LLM infer-ence.

# 3.2. Pruning

Pruning [LeCun et al., 1989, Liang et al., 2021], which con-centrates on identifying and eliminating model parametersthat are deemed unnecessary or redundant, is another pop-ular technique for compressing LLMs. In the context ofLLMs, the parameters often account for a considerable por-tion of the model size and computational demand. By care-fully pruning these parameters, it’s possible to streamlinethe model, making it more efficient without significantlycompromising its performance. Pruning methods can bebroadly classified into two categories: unstructured prun-ing and structured pruning, and we describe the researchprogress of each category in turn below.

# 3.2.1 Unstructured pruning

Unstructured pruning selectively eliminates individualweights or neurons from a model, leading to a sparser, yetmore irregularly structured network. This form of pruningexcels in ensuring model accuracy, however, the resultant ir-regularity in the weight distribution necessitates specializedhandling or software optimizations. SparseGPT [Frantarand Alistarh, 2023] is a groundbreaking one-shot pruningmethod tailored for LLMs. It tackles the pruning challengeby reconceptualizing it into a series of extensive sparse re-gression problems, efficiently solved by a newly developedsolver. Notably, SparseGPT can efficiently process a modelwith 175 billion parameters in just a few hours on a sin-gle GPU, and it can induce significant sparsity $( 5 0 \% )$ ) inLLMs without significantly sacrificing accuracy or neces-sitating fine-tuning. To tackle the challenge of reconstruc-tion cost in SparseGPT, Sun et al. [2023a] propose Wanda,which assesses the significance of each weight by evaluat-ing its magnitude and the norm of the corresponding input,significantly increasing the computational efficiency. Fur-ther, Yin et al. [2023a] design a set of non-uniform hierar-chical sparsity ratios to pay more attention to the layers withhigher outlier occurrences, thus boosting the pruning per-formance. Moreover, Considering the hardware support forunstructured pruning, Flash-LLM [Xia et al., 2023a] pro-poses an unstructured sparse matrix multiplication method,which is characterized by sparse loading and dense compu-tation, to implement the GPU Tensor Core’s sophisticatedsupport for unstructured sparsity.

# 3.2.2 Structured pruning

Structured pruning removes entire neurons or layers, result-ing in a cleaner, more regular structure. The pruned modelis generally more compatible with conventional hardware,however, the simplicity and regularity come at a cost: thisform of pruning can have a more pronounced impact onthe model’s performance, as it involves removing larger,

potentially more critical components. LLM-Pruner [Maet al., 2023] represents a pioneering approach in structuralpruning for LLMs. It employs a one-shot pruning tech-nique, which relies on first-order and estimated Hessiandata and necessitates subsequent fine-tuning using LoRA torestore the weights. This work is advantageous as it signif-icantly reduces both computational demands and memoryrequirements, while preserving the fundamental structureof LLMs. Sheared Llama [Xia et al., 2023b] proposes an-other noteworthy solution by combining targeted structuredpruning with a dynamic batch loading algorithm. First, itmeticulously prunes a source model into a desired targetarchitecture, meticulously chosen by analyzing the config-urations of the pre-trained model. Then, it enhances train-ing efficiency through the dynamic batch loading algorithm,which adjusts the proportion of training data from vari-ous domains. Compresso [Guo et al., 2023c] establishesa collaborative learning framework, where the LLM and aresource-efficient pruning algorithm work in tandem, withthe ability to prune Llama-7B to 5.4B while preserving theoriginal performance.

# 3.3. Knowledge Distillation

Knowledge distillation [Gou et al., 2021, Hinton et al.,2015] is a technique that facilitates the transfer of capa-bilities from a larger model (referred to as the “teacher”)to a smaller model (referred to as the “student”), allow-ing the smaller model to perform tasks with similar profi-ciency as the larger model but with reduced computationalresources [Gou et al., 2021, Shang et al., 2021]. For LLMcompression, there are two main categories of knowledgedistillation: white-box and black-box distillation. Withinthese categories, researchers have developed a range of dis-tillation methods tailored for LLMs, which are describedin detail below. Moreover, a more detailed and specific sur-vey regarding knowledge distillation of LLMs has also beencarried out [Xu et al., 2024].

# 3.3.1 White-Box Knowledge Distillation

In white-box distillation, the architecture and weights of theteacher model are fully accessible. This transparency allowsthe student model to learn not just the output of the teachermodel but also its internal representations and decision-making processes. MiniLLM [Gu et al., 2023] critiquesthe limitations of standard knowledge distillation objec-tives and suggests that reverse Kullback-Leibler divergenceis more effective for capturing the complexity of genera-tive tasks, which can enhance the student model’s responsequality and reliability. MiniLLM also introduces single-stepregularization, teacher-mixed sampling, and length normal-ization to address challenges in training, thus demonstratinggreat performance potential for distilling LLMs on the stan-

dard benchmarks. In contrast to MiniLLM, GKD [Agar-wal et al., 2023] presents a more straightforward and sta-ble method. It aligns more with supervised training byavoiding backpropagation through the student model’s sam-pling. Instead of using predetermined output sequences,GKD trains the student model on its own created sequences,utilizing the teacher’s probabilities as guidance, which leadsto notable improvements in the student’s performance. Ho-motopic distillation [Liang et al., 2023a] aims to facili-tate the alignment of the student model’s predictions withthose of the teacher model across extensive open-domaindata. It involves starting the student model with the teachermodel’s configuration and progressively reducing the stu-dent model’s neurons to reach a specified model complexity.Furthermore, Liang et al. [2023b] present a layerwise distil-lation approach that involves creating unique task-aware fil-ters for each layer of teacher and student models. These fil-ters, essentially neural networks equipped with task-specificheads, are designed to distill and capture the predictiveknowledge from the hidden layers of the respective mod-els. AD-KD [Wu et al., 2023b] analyzes the teacher model’stoken-level rationale using integrated gradients and trans-fers attributional knowledge to the student model, which en-ables the student model to imitate the teacher’s underlyingreasoning, not just its behaviors.

# 3.3.2 Black-Box Knowledge Distillation

Contrary to white-box distillation, black-box distillationdoes not require access to the internal information of theteacher model. Instead, it focuses on replicating the out-put behavior of the teacher model. The student modellearns solely from the input-output pairings produced bythe teacher, without any insight into its internal operations.Multitask-ICT [Huang et al., 2022] introduces in-contextlearning distillation, which merges the objectives of in-context learning with those of language modeling, intend-ing to distill into smaller models both the capability to com-prehend in-context examples and the knowledge requiredfor specific tasks. LaMini-LM [Wu et al., 2023a] creates aset of 2.58 million instructions and employs GPT-3.5 Turboto produce responses to these instructions. Subsequently,it uses these instructions as a basis to fine-tune a range ofstudent models. Similarly proceeding from creating exam-ples, Sahu et al. [2023] proposes PromptMix, which in-volves a two-step method based on prompting to create la-beled examples for text classification. In PromptMix, theborderline examples can enhance the knowledge transferfrom teacher models like GPT-3.5 to student models. Incontrast to the traditional unidirectional knowledge distilla-tion, Lion [Jiang et al., 2023] introduces an adversarial dis-tillation framework, which encourages the teacher model toidentify ”hard” instructions and subsequently generate new

”hard” instructions for the student model, resulting in a dy-namic three-step adversarial cycle.

Black-box distillation is also identified as a promis-ing tool to transfer the power of chain-of-thought (CoT)prompting from larger models to smaller ones. Fu et al.[2023b] observe a trade-off in language models betweentheir diverse capabilities, and focus on moving the teachermodel’s capability from general abilities towards enhanc-ing the student model’s proficiency in the targeted mathe-matical CoT. SCOTT [Wang et al., 2023] uses contrastivedecoding for better rationale supervision and a counterfac-tual reasoning objective for faithful distillation, resulting inmore faithful CoT rationales. Distilling step-by-step [Hsiehet al., 2023] introduces a novel training method for smallermodels, surpassing LLMs with less data. It uses LLM ratio-nales as extra training material in a multi-task framework,cutting down data needs versus standard fine-tuning and dis-tillation. Similarly, Li et al. [2023b] propose symbolic CoTdistillation, where they obtain CoT rationales for unlabeleddataset instances from the teacher model and then train thestudent model to forecast both the rationale and the labelbased on these instances. To promote complex, multi-stepreasoning within a dialogue context, i.e., dialogue CoT rea-soning, Chae et al. [2023] utilize LLMs as inconsistentteachers and strategically distill valuable and logical ratio-nales through alignment filters.

# 3.4. Factorization

The use of low-rank matrix decomposition [Kishore Kumarand Schneider, 2017] as a technique for compressing DeepNeural Networks (DNNs) represents a straightforward yeteffective approach, garnering considerable attention withinboth scientific computing and machine learning domains.In recent years, the challenge of efficiently compressing andaccelerating large-scale neural networks via low-rank meth-ods has become a focal point of research. This has led tosignificant advancements in developing and refining low-rank factorization strategies tailored for DNNs [Schotthofer¨et al., 2022].

Activation-aware Singular Value Decomposition(ASVD) [Yuan et al., 2023d] is the first work usingfactorization techniques to compress LLM. ASVD effec-tively manages activation outliers by adjusting the weightmatrix based on the activation distribution, improvingdecomposition accuracy and efficiency. ASVD alsoaddresses the varying sensitivity of different LLM layersto decomposition, with an iterative calibration processfor optimal layer-specific decomposition. Concurrently,LAyer-SElective Rank reduction (LASER) [Sharma et al.,2023] demonstrates the surprising result that it is oftenpossible to significantly improve the performance of LLMsby selectively removing higher-order components1 of theirweight matrices. Apart from targeting the LLMs’ weights,

TensorGPT [Xu et al., 2023b], in which the embeddinglayer of LLMs is compressed through Tensor-Train De-composition (TTD) [Oseledets, 2011] in order to store largeembeddings in a low-rank tensor format, with much fewerparameters.

# 4. Algorithmic Methods for Fast Decoding

LLMs have achieved astonishing performance in varioustext generation tasks. They typically contain the decoderstage that generates tokens one after another, followingan autoregressive relation with all preceding tokens. Dur-ing the decoding of every token, decoder weights have tobe repeatedly loaded into memory. As the parameter sizeof LLM becomes colossal, the decoding process becomesheavily memory-bound [de Jong et al., 2023] and experi-ences low hardware utilization, leading to severely long la-tency [Kim et al., 2023d]. This is particularly problematicin real-world applications like ChatBot, where quick andeven real-time responses are crucial. Therefore, there is astrong need to optimize the decoding process to improveperformance in such applications.

This section focuses on the discussion of prior efforts toreduce the LLM inference cost from an algorithm perspec-tive. Specifically, this section intends to develop the discus-sion from two directions:

• In section 4.1, for every single token decoded (fixed #to-kens decoded), how to utilize the minimum number ofparameters of the LLM.

• In section 4.2, for every single forward propagation of theLLM (fixed #parameters used) how to decode the maxi-mum number of tokens.

# 4.1. Minimum Parameter Used Per Token Decoded

Interestingly, Simoulin and Crabbe´ [2021] has shown thatalthough language models tend to have a huge number ofparameters, not all parameters are needed to generate theaccurate tokens. LLM inference latency can be reducedby selecting only a subset of necessary parameters to use(load) per input token and still preserving the decoded to-ken’s accuracy. In this section, we look at input-dependentdynamic weights dropping scheme for LLM from three dif-ferent perspectives: 4.1.1 looks at early exiting, or dynami-cally choosing weights in the layer, depth, dimension; 4.1.2introduces methods that dynamically detects sparsity in thewidth dimension of the LLM, pruning out heads and MLPcolumns; 4.1.3 present Mixture-of-Experts (MoE) whichpretrains a sparse model and chooses the correct experts forthe different input during runtime.

# 4.1.1 Early Exiting

Early exiting (or layer skipping) has been a well-exploredidea in various network architectures, particularly for the

encoder-only models [Baier-Reinio and Sterck, 2020, Houet al., 2020, Li et al., 2021, Liu et al., 2020, 2022, Schus-ter et al., 2021, Schwartz et al., 2020, Stickland and Mur-ray, 2019, Xin et al., 2020, Zhou et al., 2020, Zhu, 2021].Early exiting for decoder architecture requires consistencyand quality retaining on the sequence level where each to-ken depends on the previous tokens, which are considera-tions lacking in the previous abundant encoder-only early-exiting literature. The decoder contains layers of identicalstructure. Benefiting from this trait, the output hidden statesof every layer can be used to pass in the LM Head to geta probability distribution prediction of the next token de-coded. Geva et al. [2022] and Simoulin and Crabbe´ [2021]observe that for some tokens, the hidden states saturate dur-ing the intermediate layers. In other words, early exiting inthe middle would, for some tokens, output the correct top-1prediction as running through the full model. This observa-tion lays the basis for the success of decoder early exitingmethods.

Elbayad et al. [2020] conducts an early effort for effi-cient machine translation tasks to use early exiting on thedecoder architecture. It proposes a general approach to fol-low. Shown in Figure 12 (b), during the forward propaga-tion, after every layer, there is an internal confidence func-tion, usually a fixed metric or an MLP with a small numberof layers, that computes a confidence score based on thehidden states on how likely it is to saturate at the currentlayer. The score is used to decide whether to exit throughsome carefully designed criteria. The LM Head is then usedto output the next token-predicted probability distribution.Due to the high similarity of the newer follow-up works, weextend the discussion by looking at the key challenges of de-signing early exiting schemes for language models, wherethey introduce different novel techniques.

Modeling the Confidence of Saturation. CALM[Schuster et al., 2022] studies three different ways to outputthe confidence score to exit: the softmax response, or thedifference between the top two values after the softmax; thesaturation of hidden states, or the cosine similarity betweenthe current layer’s hidden states with the last layer; the out-put of a linear classifier inserted to every layer. The lin-ear classifier is trained by simply using a cross-entropy lossto align MLP output when inputting the hidden states withwhether the top-1 token decoded as exiting on the currentlayer matches the top-1 token decoded of the full model.The experiments presented suggest that despite not beingthe most accurate predictor, the classifier method reachesthe optimal trade-off between additional FLOPs overheadwith prediction accuracy on score generating. Building upfrom CALM, [Bae et al., 2023] observed that when con-sistently exiting from shallow layers will result in an ab-normally long length. Also, the confidence score computa-tion on every layer injects high overhead and diminishes the

<!-- IMAGE:14 -->


<!-- IMAGE:15 -->


<!-- IMAGE:16 -->


<!-- IMAGE:17 -->



For some tokens, only a necessary subset of weights is needed to decode correctly. (a) presents the normal decoder-only autoregressive modelarchitecture; (b) Early Exiting aims to only use the necessary transformer layers that already produce correct decoded tokens and skip the rest of thelayers, but challenges arises when different tokens exit at different layer across. Same gist also applies to the width dimension of the LLM. (c)represents a line of works that predicts the dynamic sparsity on attention head and linear layers column dimension in the input-dependent way. (d)presents the Mixture of Expert method which has the sparsity built-in to the pretrained model. During inference, a router aims to choose concise andproficient expert networks that depending on the input. (Illustration best viewed with color)



Figure 12. Illustration of Input-Dependent Dynamic Network Technique


benefit of early exiting. Therefore, it proposes to only havetwo choices for early exiting: either exit from the so-called”shallow module” or a group of shallow layers, or go all theway to the full model, or ”deep module”, drastically reduc-ing the number of classifiers needed inside the model. Suchdesign enables it to achieve more speedup than CALM,reaching $2 \mathbf { x }$ for certain tasks. On the other hand, Consis-tentEE Zeng et al. [2023] proposes a different method topredict when to exit. It uses an RL policy network that isiteratively trained with the per-layer output classifier head.The policy networks are trained with the goal of balancingthe optimization of both efficiency (the early layer receivesrewards) and accuracy (the reward function has a term thatis the early exit output CE loss).

Early Exit Criteria. CALM Schuster et al. [2022] pro-poses a distribution-free calibration technique that uses thefixed sequence testing procedure (Family-wise Error Rateprocedure) to output the suitable threshold. The thresholdis exponentially decreasing to allow more aggressive exit-ing for tokens later in the sequence. Bae et al. [2023], onthe other hand, observes that the pattern of confidence cri-teria resembles a beta distribution and uses the on-the-flydata to update a beta distribution model through MLE anduse such probability model to guide its decision. Zeng et al.[2023] bypasses this issue by letting the policy network di-rectly output the exit decision.

Hidden States Propagation. Hidden states of theskipped layers can pose a technical challenge. As shown inthe 12 (b), the token position at ”school” exits later than pre-vious tokens. However, the last self-attention layer doesn’thave the previous key-value pairs of the previous early ex-ited tokens. Elbayad et al. [2020] and Schuster et al. [2022]proposes the ”hidden states propagation” technique. For ex-ample, the hidden states of token ”Max” at the exited layer$l _ { 1 }$ are stored. When the later token ”school” reaches deeperlayer $l _ { 2 }$ , the hidden state for ”Max” is copied for all layersbetween $l _ { 1 }$ and $l _ { 2 }$ , and the key-value pairs are then com-puted on the copied hidden states. Basically, to approximatethe deep layer’s hidden state with the ones from the earlylayer. Later works Bae et al. [2023] and Ding et al. [2023]found that state propagation leads to performance degrada-tion. Since LLM inferences are dominated mostly by mem-ory loading, computation is relatively ”free”. These twomethods proposed to recompute the later hidden states di-rectly on the fly. Chen et al. [2023b] proposes to run the fulllarge model in parallel to the early exit stream to efficientlyparallel the computation of the missing kv cache. Din et al.[2023] conducts a systematic study on using a linear net-work to jump across layers for transformer architecture andshows that linear layers can be added to effectively bridgethe performance gap between directly copying and comput-ing the hidden states with low memory and compute cost.

SkipDecode Corro et al. [2023] chooses an aggressive ap-proach to prioritize the speedup and relax the performancepreservation goal. By utilizing the observation that a tokencoming later in the same sequence on average requires lessnumber of layers to decode the correct tokens, it completelybypasses the need for state propagation by forcing the max-imum layer used to be monotonically decreasing for deeperpositions. Besides, SkipDecode also introduces fixed exitpoints to optimize for batched early exit.

Output Classifier Training. When exiting from inter-mediate layers, the intermediate hidden states need to gothrough an output classifier head to output prediction ofthe next token probability distribution. The output classi-fier can either be shared as shown in Figure 12 or per-layerindependent. These classifiers are usually trained to bet-ter adapt to the early exiting pattern. Elbayad et al. [2020]proposed to have an average CE loss of all layers to be thetraining loss of the classifier. On the other hand, Schus-ter et al. [2022] uses a weighted average where weights in-crease as the layer number increases, assigning more con-tribution to deeper layers. Bae et al. [2023] introduces adynamic knowledge distillation loss which dynamically as-signs the ”shallow module” a suitable hidden state from the”deep module”. Both Rotem et al. [2023] and Ji et al. [2023]find a ”conflicting gradient” issue when joint training withthe same loss across all models: Rotem et al. [2023] de-tects the gradient conflict between early and later layers oflanguage models, while Ji et al. [2023] spots the ”orthog-onal gradient” between the objective to improve semanticawareness and the objective to improve early exciting deci-sion. Both methods propose adding an additional block ofparameters and iterative training to alleviate the issue. Be-sides the above-mentioned perspectives, Chen et al. [2023b]studies system-level optimization techniques to efficientlyrun LLM early exit under the 3D parallelism setting.

# 4.1.2 Contextual Sparsity

While early exiting aims to select parameters on the depthdimension, some techniques have also been proposed to ex-ploit the dynamic sparsity on the width dimension. DejaVu Liu et al. [2023c] conducts a comprehensive study ondynamic sparsity on the LLM width dimension. The pa-per reveals that contextual sparsity can go up as high as$80 \%$ , meaning that the majority of the weights can be leftout while still preserving the original model performance.However, the chosen weights are dynamic and different fordifferent input tokens. The paper formulates this problemas a nearly neighbor search problem that for a given hid-den state from the embedding layer of previous layers, howto find the attention heads and the MLP columns that arethe most similar to these tokens. To save compute, the pa-per proposes to train a small MLP network as the Sparse

Predictor in front of the Multi-Head Attention (MHA) andthe Feed-Forward Networks (FFN) of the LLM, shown inFigure 12 (c). By using only a subset of weights and reduc-ing the memory IO overhead, Deja Vu manages to achieveover $2 \mathbf { x }$ speedup of LLM inference. Building on Deja Vu,PowerInfer (Song et al. [2023]) brings the contextual spar-sity finding to the LLM inference across heterogeneous de-vices (CPUs and GPUs). PowerInfer discovers a substan-tial portion of weights are heavily used and activated in theinput-independent setting, thus stored on the GPU memory,while others are on the CPU memory. Then, to specificallyfind the weights to use for a given input token, it trainsa smaller sparse prediction than Deja Vu. To make thesparse predictor, it initializes the sparse predictor to havea dynamic structure and iteratively trains and modifies thesparse predictor. To better do inference of the model de-ployed on the mixed CPU and GPU environment, it intro-duces a novel memory placement scheme and implementsa vector-based sparse computation library. Concurrently,MatFormer (Devvrit et al. [2023]) studies the problem ofLLM deployment on various heterogenous devices of dif-ferent hardware capabilities. They added dynamic structureonly on the FFN, which occupies $60 \%$ of total weights. Themodel is specially trained so that during inference, basedon the target hardware properties, MLP layers are sampledon the row dimension to give a model of various sizes withreasonable performance. To diversify the model size selec-tion, it imposes a Mix’n’Match method to choose differentsettings for different layers, so combined would give a morevariable model size.

# 4.1.3 Mixture-of-Expert Models

Language Models, especially transformer architecture, ex-hibit strong power-law scaling (Kaplan et al. [2020], Hoff-mann et al. [2022]) of performance when the trainingdataset is scaled up. On the other hand, though bringingstrong performance gain, the large parameter count makestraining and inference of the model inefficient. The mixtureof expert (MoE) technique is a well-studied topic (Yukselet al. [2012]) that effectively decouples the parameter countof the model and the computation FLOPs required by themodel training and inference, thus bringing huge gains ofefficiency under certain conditions. Further, MoE is shownto effectively scale up the language model size and increaseits performance without the concern of increasing com-pute during inference (Lepikhin et al. [2020], Fedus et al.[2021]). Shown in Figure 12 (d), an expert network is in-serted into the transformer architecture to replace the FFNlayers. Also, a gating function is introduced between theMulti-Head Attention and the expert network which aims toselect the best-fit expert or experts for the given input token.For in-depth analysis and discussion about the MoE scaling

<!-- IMAGE:18 -->



Several methods have been proposed to generate more than one token per LLM forward prop: (a) shows the speculative decoding which asks the LLMto, instead of generate tokens autoregressively, evaluate tokens generated by the small model all at once to parallelize the memory loading; (b) introducesanother approach to have a last layer projection on the large model hidden state vector to generate tokens at subsequent positions; (c) presents anothermethod to treat decoding as solving a system of nonlinear equations and use Jacobi and Gaussian-Seidel iteration algorithm to parallelize the decodingprocess of a number of token positions; (d) presents the Non-Autoregressive Transformers which targets the seq-to-seq model. The decoder relaxes theautoregressive constraints and iteratively decode all the tokens from masked tokens based on the information from the encoder model hidden states.



Figure 13. Illustration of the Parallel Decoding Methods


generalization, routing algorithms, training techniques, etc.,we refer the readers to the survey on Sparse Expert Models(Fedus et al. [2022]). Although both rely on the input to-ken to determine sparse structure, We deliberately separateMoE and the contextual sparsity techniques because the lat-ter operates on pre-trained dense language models and ex-ploits the sparsity from the dense neural networks, while theprior trains a sparse model from the beginning. More re-cently, MoE techniques have achieved substantial success.Sparse Mixer (Lee-Thorp and Ainslie [2022]) brings $89 \%$and $98 \%$ speedup in both training and inference to BERT(Devlin et al. [2019]) models. Du et al. [2022] uses only$49 \%$ FLOPs but beats GPT-3 (Brown et al. [2020]) in per-formance. ST-MoE (Zoph et al. [2022]) brings MoE to theencoder-decoder models, even becoming the state-of-the-art model for many reasoning and generation tasks. ST-MoE, using $2 0 \mathrm { x }$ and 40x fewer FLOPs in training and infer-ence, beats 540B PaLM (Chowdhery et al. [2022]) in per-formance. Mixtral 8x7B (Jiang et al. [2024]), while onlyactively using 13B parameters during inference, performson par with Llama2-70B models (Touvron et al. [2023b])across various evaluation benchmarks.

Besides, various attempts have been made to optimizeMoE model inference. Kossmann et al. [2022] builds anefficient compiler library RECOMPILE for MoE modelsthat introduce dynamic recompiling and optimization ac-cording to varying inference batch sizes. Rajbhandari et al.[2022] extends the ZeRO distributed inference method toMoE models. Jawahar et al. [2023] conducts Neural Archi-tecture Search (NAS) on the expert network architecture.Yi et al. [2023] deploys large MoE language models on theedge devices. It optimizes the deployment around the find-

ing that some neurons are much more heavily used in theMoE models than others.

# 4.1.4 Roofline Model Analysis for Dynamic ParameterReducing

The Minimum Parameter Used Per Token Decoded meth-ods simultaneously decrease computational and memoryaccess overhead. From the viewpoint of roofline model,these methods result in small changes to the arithmetic in-tensity of each operator and the type of bound.

For Early Exiting or Layer Skipping methods, entireTransformer layers are skipped, leading to a proportionalreduction in overall computation, memory access, and in-ference time. In other words, the inference time decreasesproportionally to the number of layers skipped in the net-work. However, for methods like Contextual Sparsity andMixture of Experts, the arithmetic intensity varies acrossdifferent operations. Consequently, dynamically choose toactivate these layers leads to varying reductions in compu-tation and memory access, resulting in different impacts onthe overall inference time.

# 4.2. Maximum Tokens Decoded Per LLM ForwardPropagation

Another angle to reduce the latency of LLM inference is torelax the LLM from the limitation of autoregressive decod-ing and have more than one token decoded per one LLMforward propagation. We look at two ways to achieve it:4.2.1 presents the speculative decoding method which in-troduces a computationally efficient draft model to proposecandidates for the next few token positions, while the LLM

is used to evaluate the draft model’s proposed draft tokens,instead of generating next tokens. On the other hand, 4.2.2presents works that enable the LLM to directly decode mul-tiple tokens from a single forward propagation. Due to somemethods combining the benefits from both directions andlying in the middle, we manually add a distinction just forthe sense of nomenclature that speculative decoding meth-ods here all have the draft model to be in the transformerarchitecture.

# 4.2.1 Speculative Decoding

Due to the demanding memory loading challenges and au-toregressive properties, LLMs are inefficient in inference.However, models that are much smaller in size are shown(Kim et al. [2023e]) to have the ability to decode the cor-rect sequences as the LLM, as long as some key tokens inthe sequence of the small model generation are corrected.Then, shown in Figure 13 (a), when the small model isasked to infer (speculate) and output a sequence of drafttokens, memory loading of model weights is less of a prob-lem, resulting in much higher utilization in hardware com-putation units. To ensure the quality of the text generated bythe small model, the LLM can ”periodically” evaluate andcorrect tokens from the small model’s draft. Then, althoughthe large model needs to sometimes evaluate the wrong drafttokens, potentially leading to larger FLOPs spent than LLMautoregressive decoding, the memory loading of weights isparallelized on the token dimension and drastically reducesthe memory IO overhead. Since the LLM inference is mem-ory bottlenecked, the speculative decoding will potentiallyreduce the LLM inference latency greatly.

LLM Distribution Preserving During early explorationof this idea, two different paths emerged concurrently. Kimet al. [2023e] proposed to have the small model speculateand generate draft tokens until the token decoded confi-dence falls below a threshold. Then, the small model ”fall-back” to the large model to evaluate the draft tokens gen-erated and hand over to the small model. Some of the to-kens are rejected, so the large model asks the small modelto ”roll back” these wrong tokens and resume speculating.In the paper’s setting, all decoding is ”greedy”. The papershow that the large and small model pair can generate textwith quality on par with the original large model autore-gressive generated text. However, Leviathan et al. [2023]and Chen et al. [2023a], upon the small model speculateparadigm, points out a technique of resampling that at theposition where the LLM rejects the small model’s predic-tion that provably enables the large and the small modelpredictions to be in the same probability distribution asthe large model’s autoregressive generation. The follow-ing techniques generally follow the paradigm of speculatingthen evaluating and resampling to preserve the LLM autore-

gressive decoding quality while enabling speedup.

Building a Tree of Draft Tokens Since the LLM gen-erates in the autoregressive order, every token is dependenton all previous tokens generated, and the length of the ac-cepted tokens in the small model’s draft is usually modestand bounded. It is exponentially more difficult to specu-late on tokens more distant in the future. For example, ifthe small model is asked to output the length m draft se-quence, and the LLM accepts n, $\mathrm { ~ n ~ } < \mathrm { ~ m ~ }$ , the (m - n) to-kens are automatically discarded. Thus, the speedup ratio ofspeculative decoding is modest, since every LLM forwardleads to only a limited number of tokens being decoded.There are two ways to improve the speedup of speculativedecoding. First, Sun et al. [2023b], Miao et al. [2023b],and Xu et al. [2023a] all proposed to boost the draft onthe batch size direction, or letting the small model samplemultiple plausible draft sequences for the LLM to evalu-ate in parallel. Specifically, Sun et al. [2023b] proposes away and theoretical guarantees for the LLMs to batch ver-ify and resample from the multiple small model drafts sothat the LLM distribution is preserved and no loss of gen-eration quality is incurred. The paper first connects specu-lative decoding to the broader problem of discrete optimaltransport. The small model is asked to sample multiple draftsequences using topk sampling. Based on the properties ofthe discrete optimal transport, finding the optimal methodto evaluate and resample becomes finding the optimal trans-port path. On the other hand, besides from maintaining thespeculative decoding consistency of draft trees, Miao et al.[2023b] constructs the token tree not based on the top pre-dictions from the small draft model, but based on multiplediversely trained small draft models, each running in par-allel and output diverse but powerful draft sequences. Thepaper proposes a novel draft token tree construction algo-rithm that builds a tree of candidate tokens based on thediverse draft sequences through predefined expanding andmerging schemes. Then, the large model is asked to paral-lel verify the constructed tree using a carefully designed treeattention to maximize the reuse of the key-value cache andmaintain a tree-based causal mask. Xu et al. [2023a] inno-vatively applies the benefit of speculative decoding to edgedevices. The paper builds an LLM serving engine for theedge, where a smaller draft LLM is sitting consistently inmemory, while a larger robust LLM is occasionally loadedin memory to do verification. To boost the acceptance ratefrom the large LLM, it also constructs a tree using topk to-kens. To cater to the edge hardware characteristics, it im-plements a tree-based parallel verification decoder equippedwith masking and a customized large-small LLM computa-tion pipeline to avoid memory contention.

Knowledge Distillation and Self-Speculative Decod-ing Another way to improve the acceptance rate is to im-prove the small draft model’s ability to align with the LLM’s

generation distribution, which can be done through finetun-ing the small models on corpus generated by the large mod-els with knowledge distillation. Zhou et al. [2023c] estab-lishes a mathematical connection between the acceptancerate and natural divergence between the small model andthe LLM: minimizing the divergence is maximizing the ac-ceptance rate. The paper also studies a range of differentknowledge distillation losses and shows that adding knowl-edge distillation brings consistent $10 \%$ improvement inlatency speedup. However, the paper generally finds that theoptimal knowledge distillation loss choices vary model bymodel and should be tuned as a hyperparameter. Liu et al.[2023a] also shows that knowledge distillation boosts thesmall model training. Besides, the paper brings speculativedecoding to the cloud online learning settings. LLM infer-ence is memory-bottlenecked, which means that there is al-ways a surplus in computation resources. The compute canbe used to train a draft model continously on server, whichbrings two benefits: Continuously training with knowledgedistillation boosts its acceptance rate and, thus, reducesthe LLM inference latency; 2) serving input is constantlyshifting in domains, and continuous training helps the draftmodels maintain the strong performance in different do-mains. Zhang et al. [2023] avoids storing a separate draftmodel by selectively sampling a smaller draft model fromthe large model itself. Before deployment, the paper utilizesa Bayesian optimization method to search for a draft modelby skipping intermediate layers within the pretrained largemodel. Besides, it proposes an adaptive threshold selectiontechnique tailored for the decoding of the draft model sam-pled from the large models.

# 4.2.2 Parallel Decoding

Alternatively, abundant works have been proposed to en-able the large model to directly perform parallel decodingwithout the help of a small transformer model.

Simultaneously Predicting Multiple Future Tokens Awide variety of works are exploring the subject of enablingmultiple token predictions directly from one forward passof the Large Language Model. Stern et al. [2018] pioneersthe design of inserting a linear projecting layer between thelast hidden states output and the input of the language mod-eling head to enable multiple future tokens to be projectedsolely based on the current token’s last hidden states as in-put. Evaluation is subsequently made by the LLM to decidewhether to accept or reject these projected tokens. The pro-posed technique focuses on the sequence-to-sequence mod-els that have the decoder structure. More recently, Cai et al.[2024] extends the previous work to the decoder-only lan-guage models as shown in Figure 13 (b). Besides the lastlayer projection, to further improve the decoded acceptancerate, the paper proposes to add a tree-based decoding struc-

ture and the associate attention mask design to propose mul-tiple drafts simultaneously for the large model to evaluate.Besides, concurrently Monea et al. [2023] proposes to addseveral dummy tokens at the end of the input sequence arecalled ”lookahead embeddings” in work. During the for-ward pass of each layer, the information of previous prompttokens and already decoded tokens can be used to paral-lel decode several consecutive future tokens. To enablethis design, the work trains a separate embedding layer thatspecifically serves these lookahead embeddings. Li et al.[2024] also aims to do parallel decoding with LLM evalua-tion. Like previous works, it also adds a lightweight struc-ture FeatExtrapolator. Differently, the structure takes boththe previous token’s last layer hidden states and the actualdecoded token embedding as input and output the hiddenstates prediction of the next layer. The LM head of the LLMis used, and several tokens are sampled, which are then usedto build a decoding tree for the LLM to evaluate in parallel.

Retrieval of Frequent N-grams Besides directly usingthe LLM to output several following tokens, some worksuse the frequently appeared n-grams in natural language toenable multiple future tokens to be generated within oneforward pass of the large model. LLMA (Yang et al. [2023])first observes that the generation tasks tend to ask the LLMto repeat tokens that appeared in the previous contexts.Based on this information, the paper set out to use the de-coded tokens and the prompt to do prefix matching with aset of reference documents so that if a repetition occurs, to-kens that are repeated can be directly copied to the currentplace. Then, an LLM will evaluate these found candidatetokens from the previous context to decide whether to usethem. He et al. [2023] further extends LLMA and proposesto first construct a database of common phrases based onthe LLM pretrained or finetuned dataset and corpus. Then,during decoding, the previous context prompts or tokens areused as the query to be used to retrieve into the constructeddatabase. The candidates retrieved are organized into a pre-fix tree structure or a trie, which the LLM can then evaluateefficiently. Lan et al. [2023] similarly follows to use the re-trieval methods to speed up inference. In contrast, it adds anextra attention layer at the end of the LLM to use the currentcontext represented by the hidden states of the current tokenas the query to attend to relevant phrases retrieved from thedocuments of reference and select top phrases based on theattention scores.

Hierarchical Structure In Language HierarchicalStructure exists in language. For writing a long piece ofarticle, the usual approach is to first write out the generaloutline of the paper, as in the format of bulletin points.Then, for every bulletin point, arguments can be extendedto encapsulate the full intent of the bulletin point. Based onthe observation that arguments for different bulletin pointsare relatively independent in semantics, some methods are

proposed to parallelize the generation process for differentbulletin points. Skeleton-of-Thoughts (Ning et al. [2023])proposed to first ask the LLM to generate concise bulletinpoints for an article, and then collect these bulletin points onthe batch axis and feed them into the LLM again as a promptto ask the LLM to expand the arguments for each bulletinpoints in parallel. The achieved speedup is approximately2x, but with the caveat that the method cannot easily gen-eralize to all text generation tasks. More recently, APAR(Liu et al. [2024a]) extends upon this direction. The pa-per adds specific soft tokens that explicitly inform the LLMof the hierarchical information during the generation. TheLLM is further instruct-tuned to incorporate the added spe-cial tokens, and the generation is boosted with the Medusa(Cai et al. [2024]) technique to achieve 4x speedup on textgeneration with the hierarchical structure.

Jacobi and Gaussian-Seidel Iterative AlgorithmsSong et al. [2021] pioneers the study of using parallelizablemethods to approximate the results from iterative and se-quential inferences of fully connected networks or CNNs.Though seemingly in-viable, the paper finds that neuralnetworks can tolerate numerical approximation errors andthe data patterns that neural networks learn expose paral-lel structures to some extent, which makes it possible insome scenarios to parallelize the sequential inference ofneural networks. Jacobi and Gaussian-Seidel Algorithmswere previously proposed to solve a system of non-linearequations (Ortega and Rheinboldt [2000]) and are shown toeffectively parallelize the sequential neural network infer-ence. Santilli et al. [2023] extends the Jacobi and Gaussian-Seidel algorithms to parallelize the autoregressive decodingin the Machine Translation tasks. Specifically, this work isbuilt on top of the previous Non-Autoregressive Transform-ers architecture (which we will cover later in the chapter) toenhance the parallel decoding with GS-Jacobi algorithms.The parallel decoding process stops when a [EOS] token isfound in the decoded text. Concurrently, Lookahead decod-ing (Fu et al. [2023a]) shown in Figure 13 (c) extends thismethod to parallelize the LLM generation of subsequent to-kens. Besides using the vanilla Jacobi iterative algorithm,it also boosts its speed with a retrieval-based algorithm toreuse the previously seen n-grams. In addition, it paral-lelizes the lookahead step and LLM verification step by in-troducing a carefully designed attention mask to the originalLLM model to further improve decoding efficiency.

Non-Autoregressive Transformers For Machine Trans-lation tasks that require autoregressive decoding of thesequence-to-sequence model, Non-Autoregressive Trans-formers (NAT) has been proposed to iteratively decode allof the output tokens together, as shown in Figure 13 (d).NAT has been relatively well-explored (Gu et al. [2017],Wang et al. [2019], Li et al. [2019], Sun et al. [2019b],Wei et al. [2019], Shao et al. [2020], Lee et al. [2018],

Ghazvininejad et al. [2019], Guo et al. [2020], Gu and Kong[2020], Savinov et al. [2021]), and we point the readersto the following survey paper that covers specifically NATmodels Xiao et al. [2023c] for an in-depth review and anal-ysis on the subject. Coarsely, the speedup of text decodingcomes from making a single forward pass of the decoderoutput more than one token. The input sequence is first fedinto the encoder, which outputs the hidden states that extractthe input semantics. The output hidden states of the encoderare then used as the condition for the decoder pass. To speedup the text generation, the decoder side relaxes the autore-gressive constraints and takes a sequence full of dummy to-kens [pad] as the input to start the iterative parallel decodingprocess. During each iteration, based on the condition setby the encoder output hidden states, some tokens can beconfidently predicted, which are unmasked. The sequenceis mixed with unmasked decoded tokens and the remainingmasked tokens are fed to the decoder again until every tokenis decoded. The length of the sequence fed into the decoder,or fertility, is usually learned either inside the encoder as aspecial [CLS] token or by a specialized fertility predictorbetween the encoder and the decoder. More recently, Savi-nov et al. [2021] treats the decoder as a diffusion model andtrains it to denoise the noisy initial sequence based on theconditions given. However, because of the requirement touse encoder hidden states as the condition for parallel de-coding, NAT methods face natural difficulties in extendingdirectly to decoder-only architectures.

# 5. Compiler/System Optimization

After model compression and algorithm optimization forLLMs, the next step is to compile and deploy them on hard-ware devices. To ensure efficient inference of LLMs, thereare various compiler optimizations that can be employed.Moreover, due to the increasing scale of LLMs, multiplehardware devices may be required for deployment and exe-cution, forming a complex inference infrastructure system.As a result, system-level optimization for efficient inferencehas become a hot topic. In this section, we will exploresome widely used compiler optimization and system opti-mization techniques. These include operator fusion, mem-ory management, workload offloading, and parallel serving.

# 5.1. Operator Fusion

Operator fusion is an important compile-time optimizationtechnique in deep learning frameworks to improve com-putational efficiency. It combines together multiple opera-tors or layers that are directly connected in the computationgraph. This eliminates redundant data movement and in-termediate representations. For example, a linear operatorfollowed by a SiLU operator can be fused together into asingle operator. As shown in Figure 14, this avoids havingto store and load the intermediate activations between each

<!-- IMAGE:19 -->



Figure 14. Demonstration of operator fusion for a linear operatorfollowed by a SiLU operator.


<!-- IMAGE:20 -->


<!-- IMAGE:21 -->



Figure 15. Demonstration of the memory-bound case andcompute-bound case for operator fusion.


<!-- IMAGE:22 -->



Figure 16. The memory access reduction and inference time re-duction for FlashAttention on Nvidia A6000.


operator, reducing both the memory consumption and thememory access. As shown in Figure 15, the roofline modelsuggests that kernel fusion can increase arithmetic inten-sity and enhance inference performance in memory-boundareas. However, when operators are already in a compute-

bound area, memory fusion provides little benefit.

While operator fusion can provide significant perfor-mance benefits in many cases, it is not applicable to all oper-ators. Operator fusion may not be possible or beneficial forcertain operators: (1) Operator fusion requires that the inter-mediate results of the fused operations are not needed else-where in the computation graph. If a subsequent operationdepends on the output of an intermediate operation, fusionis not possible without introducing additional complexity orrecomputation. (2) Operator fusion can potentially increasethe on-chip buffer requirements of the fused operation. Ifthe available on-chip buffer is limited, it may not be fea-sible to fuse certain operations. (3) Some frameworks orhardware architectures may have limitations or restrictionson which operations can be fused together, depending ontheir implementation details.

Some compilation tools, such as TVM [Chen et al.,2018], are capable of identifying operators that can be fusedtogether and replacing them with a fused operator. How-ever, for LLMs, automatically detecting and fusing opera-tors is both unnecessary and complex because LLMs havea fixed architecture. Instead, specific fusion patterns canbe used to improve efficiency. For instance, the attentionmechanism is an essential part of LLMs. Automatically fus-ing attention mechanism can be a complex task for compila-tion tools. FlashAttention [Dao, 2023, Dao et al., 2022] andFlash-Decoding [Dao et al., 2023] proposed fusing the ma-trix multiplications and softmax operator in self-attentioninto one operator. This fusion technique eliminates the needto store and load the intermediate attention matrix, whichcan be very large when the sequence length or batchsize islarge. As shown in Figure 16, fusing them can significantlydecrease the memory access and inference time. We can ob-serve that there are differences between the prefill stage anddecode stage. In the decode stage, the memory access re-duction is the same as inference time reduction. However,in the prefill stage, inference time reduction is lower thanmemory access reduction. This is because some operationsin the prefill stage are compute-bound, so reducing memoryaccess by operator fusion provides little benefit.

DeepSpeed-inference [Aminabadi et al., 2022] intro-duces a technique called Deep-Fusion. It specifically fusesfour main regions within a transformer layer: the QKVGeMM and input layer normalization; transposition and at-tention operations; post-attention layer normalization andintermediate GeMM; bias addition and residual addition.xFormers [Lefaudeux et al., 2022] offers various fused ker-nels that can enhance the performance of transformers.These include fused softmax, fused linear layer, fused layernorm, and fused SwiGLU. TensorRT-LLM [Vaidya et al.,2023] is another framework that offers a wide range ofhigh-performance fused kernels. It incorporates a powerfulpattern-matching algorithm that can detect potential fusions

<!-- IMAGE:23 -->



Figure 17. In typical computer architectures, the memory systemconsists of different types of memory spaces.


<!-- IMAGE:24 -->



Figure 18. Roofline model for different offload settings.


in various LLMs.

In addition to kernel fusion, we can enhance the per-formance of the LLM by further optimizing operators’implementation. For example, FlashDecoding $^ { + + }$ [Honget al., 2023] proposes using asynchronized softmax and flatGEMM optimization with double buffering to improve effi-ciency.

# 5.2. Memory Management and Workload Offload-ing

When using an LLM to generate responses, the number ofinput and output tokens can change each time. The lengthof the user’s input prompt may vary, affecting the length ofthe sequence in the prefill phase. Additionally, the sequencelength increases incrementally during the decode phase astokens are generated. This means that the shapes of the ac-tivations are not fixed like in a normal neural network. Howto manage the memory efficiently as the tensor sizes changeis a problem. PagedAttention [Kwon et al., 2023] efficientlyhandles the KV cache by dividing it into blocks. The KVcache of each sequence is divided into blocks, with eachblock containing the keys and values for a fixed number oftokens. To manage these blocks, a table is used to map thelogical blocks of a sequence to the physical blocks in GPUmemory. This mapping is similar to how virtual memoryworks in a CPU’s memory management system.

When the GPU has limited memory capacity and the net-work is too large to fit, it may be necessary to employ work-

load offloading to store the network in alternative mem-ory spaces. As depicted in Figure 17, a computer systemconsists of various memory spaces, including CPU’s DDR,GPU’s GDDR/HBM, and hard disk. However, these differ-ent memory spaces have distinct access bandwidths. Fig-ure 18 illustrates that when the data is offloaded to CPU’sDDR and transferred to the GPU for computation whenneeded, it is better than performing the computation onthe CPU. When the batch size is large enough, the arith-metic intensity increases significantly, allowing the GPU tofully utilize its computation capacity and achieve good re-sults. DeepSpeed-inference [Aminabadi et al., 2022] intro-duces ZeRO-Inference, which offloads the weights of largemodels to CPU memory. This mechanism performs wellwith large batch sizes because the increased batch size in-crease the computation requirement and make the computa-tion latency overlap the latency of fetching model weights,thereby improving overall efficiency. Huggingface Acceler-ate [HuggingFace, 2022] can also move certain modules tothe CPU or disk if there is not enough GPU space to storethe entire model. FlexGen [Sheng et al., 2023] provides away to explore different ways of offloading computationsconsidering constraints imposed by available hardware re-sources from the GPU, CPU, and disk. To find the beststrategy in terms of throughput, FlexGen employs a lin-ear programming-based search algorithm. Alizadeh et al.[2023] takes advantage of the larger capacity of flash mem-ory compared to DRAM. It efficiently performs inferenceby storing model parameters in flash memory and transfer-ring them to DRAM when needed.

# 5.3. Parallel Serving

Parallel serving handles multiple user requests to a serverat the same time. One goal is to respond to each requestquickly. To achieve this, we need to reduce the time it takesto respond to each user, known as the response latency. An-other important factor to consider is throughput, which isthe number of requests the server can process in a giventime. By increasing the server’s throughput capacity, wecan serve more users simultaneously, leading to better over-all system performance. By increasing the server’s through-put capacity, more users can be served simultaneously, re-sulting in improved system performance. The serving sys-tem should be optimized to maximize throughput, whilestill ensuring that the response latency is within accept-able limits. Batching is a fundamental approach to improvethroughput by processing multiple user requests together.Figure 19 shows that increasing the batch size during thedecode stage significantly enhances throughput. However,increasing batch size can increase the response latency andmemory consumption.

Several techniques have been proposed to optimize thebatching method. For example, ORCA [Yu et al., 2022]

<!-- IMAGE:25 -->


<!-- IMAGE:26 -->


<!-- IMAGE:27 -->



Figure 19. The parallel serving settings have an impact on thethroughput, latency, and memory usage of the Nvidia A6000 GPU(Llama-2-13b).


introduces continuous batching (also known as iterativeor rolling batching) to combine inferences from differentusers. SARATHI [Agrawal et al., 2023] employs chunked-prefills and decode-maximal batching. It combines prefillchunks and decode requests to create batches, which in-creases the arithmetic intensity and improves throughput.Similarly, DeepSpeed-FastGen [Holmes et al., 2024] andLightLLM [ModelTC, 2024] also employ a split and fusetechnique.

# 6. Hardware Optimization

Designing hardware to efficiently support inference forLLMs is a challenging task due to the varying arithmeticintensity5 under different inference stages and workloadconditions. Specifically, the prefill stage usually leveragesGEMM operators to process the batched tokens, which ex-hibits high arithmetic intensity. On the contrary, the decod-ing stage calculates output tokens one at a time, which ne-cessitates the use of either GEMV operators or lean GEMMoperators to process the attention and FFN layers. Theseoperators are characterized by low arithmetic intensity.

Furthermore, the arithmetic intensity can exhibit sub-stantial variation depending on the batch sizes and sequencelengths. For instance, a large batch size could significantly

<!-- IMAGE:28 -->


<!-- IMAGE:29 -->



Figure 20. Effect of bandwidth on roofline model and Llama-2-13b. (Batch siz ${ \boldsymbol { \mathbf { \rho } } } { \boldsymbol { \mathbf { \rho } } } { \boldsymbol { \mathbf { \rho } } } { \boldsymbol { \mathbf { \rho } } } { \boldsymbol { \mathbf { \rho } } } { \boldsymbol { \mathbf { \rho } } } { \boldsymbol { \mathbf { \rho } } } { \boldsymbol { \mathbf { \rho } } } { \boldsymbol { \mathbf { \rho } } }$ , sequence length $\scriptstyle 1 = 1 0 2 4$ )


alter the arithmetic intensity, and a long sequence lengthmay increase the memory access overhead of KV-cachereading in each decoding step. This variability introducesadditional complexity into the hardware design process, asdifferent stages or configurations may necessitate distinctoptimization strategies. Hence, it’s crucial to consider thesefactors when designing hardware to ensure efficient perfor-mance across a wide range of scenarios.

Considering these challenges, careful consideration andoptimization of hardware designs are necessary. In this sec-tion, we will survey and analyze various hardware optimiza-tions tailored for efficient LLM inference, with a focus onaddressing the issues related to varying arithmetic intensity.

# 6.1. Spatial Architecture

The decoding process of LLM involves predicting wordsone at a time based on previously generated ones. How-ever, this process can be costly, especially during tasks inlong sequence generation. This is because the model needsto access large amount of weights and the key-value (KV)cache to generate each token, resulting in low arithmetic in-tensity.

There are several solutions that have been developed toaddress this issue. One such solution is the implementationof a ”Spatial Architecture”. In contrast to traditional com-puter architectures, spatial architectures utilize a differentapproach to computing. Instead of folding the computa-tion process into multiple interactions between processingelements (PEs) and main memory, spatial architectures dis-tribute the computation across multiple PEs. This designallows for the exploitation of parallelism, as each PE simul-taneously performs a portion of the computation. Addition-ally, the intermediate data flows between the PEs, avoidingwriting back to DRAM each time.

In a spatial architecture, each PE is responsible for a spe-cific portion of the computation. To facilitate efficient com-munication, data is typically moved between neighboringPEs. This allows for improved performance and efficientutilization of resources. In a spatial setup, each PE has itsown direct access to memory. This enables multiple pro-cessing units to access memory simultaneously, improving

the overall speed at which information can move in and outof memory. This results in enhanced memory bandwidthand overall LLM inference performance. As shown in Fig-ure 20, as the total memory bandwidth increase, the per-formance for the linear layer in decoding stage can signifi-cantly increase.

In one case, Groq employs their LPU [Abts et al., 2022]to create a spatial system for LLM inference. This systemachieves a remarkable speed of over 300 tokens per secondper user on the Llama-2-70b model [Groq, 2023]. Anotherexample is Graphcore’s Intelligence Processing Unit (IPU),which is another type of spatial architecture that efficientlyexecutes LLMs [Graphcore, 2024].

# 6.2. Processing in Memory

The decoding phase of LLM inference experiences the so-called ”Memory Wall” problem, primarily due to its lowarithmetic intensity. This issue is not new, the computer ar-chitecture community has been struggling with the ”Mem-ory Wall” problem for several decades. Among various po-tential solutions, processing in-memory techniques (PIM)have garnered significant interest in recent years. By plac-ing compute units directly in memory chips, we can lever-age the much higher internal memory bandwidth and re-duce the data movement overhead between memory andCPU/GPU cores.

In recent years, DRAM-based PIM has entered the com-mercialization phase, which can potentially mitigate thememory bandwidth bottleneck of LLM inference. As listedin Table 2, UPMEM’s PIM-DIMM Devaux [2019a] is thefirst commercialized DRAM-PIM product, which placesgeneral-purpose RISC cores in DDR4-DIMMs. However,this product was not intended for deep-learning applica-tions, thus the peak bandwidth and throughput can hardlymeet the requirements of LLM inference. Compared toUPMEM’s PIM-DIMM, Samsung proposes to place MACunits into HBM memory, achieving 2TB/s of internal mem-ory bandwidth, which is much higher than that of traditionalHBM2 (307GB/s per cube) memory. Since the process-ing units are tailored for deep learning workloads, the peakcompute throughput of HBM-PIM can reach 1.2TFLOPS.In other words, HBM-PIM is suitable for accelerating oper-ators with 1-2 Ops/Byte of arithmetic intensity.

Choi et al has proposed to accelerate the KV-Cache pro-cessing using HBM-PIM, which has low arithmetic inten-sity in batched LLM inference. According to their eval-uation Choi et al. [2023], A GPU+HBM-PIM system forLLM inference can achieve $3 . 2 4 \times$ speedup over traditionalmonolithic GPU system.

Similar to Samsung’s HBM-PIM, SK-hynix has alsoproposed a GDDR6-based PIM accelerator Kwon et al.[2022] called AiM. As shown in Table 2, the compute unitsof AiM adopts the BF16 data format, which is more efficient

for deep-learning acceleration. With optimized MAC units,AiM offers 1TFLOPS per chip of compute capacity, whilethe peak bandwidth is 1TB/s per chip. Although AiM hasnot reported its performance on LLM, it can achieve up to$1 0 \times$ speed improvement compared to GPU+HBM2 systemon LSTM tasks.

Note that, although DRAM-based PIM techniques havedemonstrated promising potential of accelerating memory-intensive operators in LLM inference, there are still somelimitations that should be addressed in the future.

• Limited Computation Power. A key restriction ofDRAM-PIMs in accelerating LLMs stems from their con-strained computational capabilities. DRAM-PIMs utilizecompute units crafted using the DRAM process, result-ing in transistors that are 3 times slower and logic den-sity several times lower compared to CMOS in the sametechnology node Devaux [2019b]. Even worse, DRAMchips usually have fewer metal layers, leading to a lowerrouting density at the same time. Due to these technicalconstraints, DRAM-PIMs can hardly incorporate power-ful compute units. As a consequence, DRAM-PIMs isonly suitable for small-batch inference or KV-cache pro-cessing. For computation-intensive large-batch inference,a powerful host is still necessary.

• Capacity Constraints. Another significant limitationof DRAM-PIMs is their restricted capacity. Given thatDRAM-PIMs allocate some memory capacity to con-struct the computation units, the total memory capacityis typically $50 \%$ less than that of standard memory Kwonet al. [2021]. For LLM applications that necessitate sub-stantial memory capacity to accommodate the weightsand KV-cache, DRAM-PIMs may encounter capacity-related challenges.

• Inadequate Inter-PIM Communication. In addition tothe constraints imposed by computation power and capac-ity, another limitation of DRAM-PIMs is their suboptimalinter-PIM communication capability. Given that thereare distributed computing units located near each DRAMbank, data aggregation and computation synchronizationamong these units are unavoidable. However, DRAM-PIMs lack robust interconnects Jonatan et al. [2024],Zhou et al. [2023d], and they typically depend on the hostCPU/GPU for data exchange between PIM units. This re-liance can lead to inefficiencies in the system. Therefore,to enhance LLM inference, future iterations of DRAM-PIMs should aim to improve their inter-PIM communica-tion abilities.

# 6.3. New Data Format

Neural networks typically employ high-precision floating-point numbers (16 or 32 bits) for training. While high-precision FP numbers can accommodate both representa-tion precision and range, the complex hardware implemen-


Table 2. Comparison of Commodity DRAM NMC Products


<!-- TABLE:2 -->

tation required for FP arithmetic is not conducive to effi-cient inference. To mitigate the hardware overhead, uni-form quantization casts high-precision floating-point num-bers to low-precision integer representations, substitutingexpensive floating-point logic with efficient integer logic.However, uniform quantization struggles to balance rep-resentation precision and range simultaneously, leading tosubstantial degradation in model accuracy. Moreover, pre-serving model accuracy without degradation necessitateswell-designed quantization algorithms, introducing addi-tional casting efforts. Non-uniform quantization attemptsto enhance the precision of data representation under low-bit conditions by assigning bits and discretizing the rangeof parameters non-uniformly. Yet a critical drawback ofnon-uniform quantization is its challenging deployment ongeneral computation hardware, e.g. CPUs and GPUs [Gho-lami et al., 2022]. In summary, existing data formats fail toconcurrently achieve fine precision, extensive range, highefficiency, and low tuning costs for inference. Given thecriticality of reducing the deployment costs of LLMs, a sub-stantial amount of work investigate into exploring the mostbalanced data format tailored to LLMs.

<!-- TABLE:3 -->


Table 3. Comparison of Floating Point, Uniform Quantization, andNon-Uniform Quantization


To trade for better hardware efficiency from the origi-nal FP model, a natural progression is to reduce the expo-nent & mantissa bits in the high-resolution floating-pointformat. As demonstrated by recent works [Micikeviciuset al., 2022, Sun et al., 2019a], models of various cate-gories pre-trained in FP16 (including LLMs) can be di-rectly quantized to FP8 without significant accuracy degra-dation. Moreover, on a wide spectrum of tasks, trainingwith FP8 can effectively match the result quality achievedby 16-bit training sessions. The significant gain of hard-ware efficiency and little demand of user efforts envisionedby low-resolution float-point formats have garnered the at-tention from AI hardware manufacturers. For instance,NVIDIA implements FP8 Tensor Core in their latest H100

GPUs [NVIDIA, 2022]. Tesla also introduces configurablefloating point format, namely CFloat8, in their Tesla Dojochip [Tesla, 2023].

Beyond the noval architectures introduced by the indus-try, academia has also initiated efforts to exploiting thepotential of low-precision floating-point format on LLMs.ZeroQuant-FP [Wu et al., 2023d] proposes FP4 and FP8for weight/activation quantization of LLMs. The authorsadopt scaling constraints for weight quantization, achiev-ing efficient weight conversion from FP4 to FP8 and betterutilization of the FP8 Tensor Core. ZeroQuant- $( 4 + 2 )$ [Wuet al., 2023c] and FP6-LLM [Xia et al., 2024] proposesweight quantization of LLMs with FP6, and provides effi-cient implementation on CUDA Core and Tensor Core, re-spectively. LLM-FP4 [Liu et al., 2023b] proposes quantiz-ing both weights and activations of LLMs down to FP4. Insummary, these efforts demonstrates the feasibility of ap-plying floating-point formats at even lower bit-widths forquantization, as well as the potential for achieving greaterefficiency gains on existing or new hardware platforms.

On the other hand, researchers are delving into the re-finement of low-precision quantization formats to augmentthe adaptability of data representation whilst preservinghardware efficiency. One line of works propose to explorenew encoding schemes within single value representation.Contrary to INT and FP numbers, which utilize fixed-lengthsub-fields for encoding distinct pieces of information, suchas exponents and mantissas, the newly-proposed rule-basedquantization formats enable dynamic adjustment of sub-field bit-widths. ALPS [Langroudi et al., 2021] proposesa generalized posit format along with a new adaptive quan-tization algorithm to optimally represent the dynamic rangeand distribution of DNN parameters. ANT [Guo et al.,2022a] proposes a new data format called flint with leading-1 encoding for exponent fields. Dybit [Zhou et al., 2023a]proposes to separate exponent and mantissa fields with thefirst encountered 0 as delimiter. The flexibility of thesevariable-length data formats presents the chance to com-promise between range and precision more effectively andallows for customization to more closely align with the dis-tribution of LLMs’ weights and activations.

Another line of works exploit similarity and discrep-ancy across values. Outlier-aware quantization capitalizeson the observation that values with large magnitudes havea significant impact on model performance. In this ap-proach, important values are identified as outliers and aretreated differently from normal values to ensure a moreaccurate representation. OLAccel [Park et al., 2018] andGOBO [Zadeh et al., 2020] separately store and allocatehigher bit-width to the outlier values. OliVe [Guo et al.,2023a] refines the concept with outlier-victim pair encod-ing scheme to ensure aligned memory access and improvedefficiency. Bit-sharing encoding concentrates on the inher-

ent similarity among values and annotates additional infor-mation in coarse granularity, thereby achieving a balancebetween representation accuracy and hardware efficiency.AdaptivFloat [Tambe et al., 2020] proposes to optimallyshift the available range of FP values with a common tensor-wise exponent bias. MX [Darvish Rouhani et al., 2023] ex-tends AdaptivFloat’s observation into finer granularity, andproposes Block Data Representation (BDR) framework toexplore the optimal tradeoff between representation accu-racy and hardware efficiency.

<!-- TABLE:4 -->


Table 4. Summary of improvements in data formats. The benefitscomes with negligible negative impacts on other factors.


# 6.4. New Processing Element

Except for the high demand for memory access, there hasbeen a growing interest in developing specialized process-ing elements (PEs) to boost the computation. These spe-cialized architectures aim to provide significant computa-tional enhancements over general-purpose processing ele-ment, such as CUDA core, for the specific operations asso-ciated with LLMs.

NVIDIA has developed a special hardware accelerationengine called the Transformer Engine in their H100 GPU.This engine uses statistical analysis to determine the opti-mal precision (FP16 or FP8) for each layer of the model,achieving the best performance while maintaining accuracy.Some researchers have designed accelerators specificallyfor efficiently executing the attention mechanism in lan-guage models (LLMs) [Kao et al., 2023, Qin et al., 2023].Several companies and research groups have been exploringthe use of FPGAs to speed up LLM computations. Exam-ples include DFX [Hong et al., 2022] and LightLLM [Zenget al., 2024].

# 7. Discussion

# 7.1. Reliability

The discussion above significantly enhances the inferenceand training efficiency of LLMs in practical scenarios.However, these compression methods also lead to subtlechanges in model reliability. Overall, the various compres-sion techniques discussed in Section 3 will have a signif-icant impact on model reliability. Therefore, this sectionprimarily focuses on how key design choices within thesedifferent compression techniques affect the following threeaspects of reliability: hallucination, safety alignment, andout-of-distribution generalization.

Hallucination primarily refers to cases where the out-puts of LLMs do not align with real-world knowledge, oftengenerating content that is either factually incorrect or non-sensical Huang et al. [2023a]. Safety alignment focuses onthe model’s inherent ability to autonomously recognize andrefuse to respond to harmful inquiries, thereby safeguard-ing against generating inappropriate or dangerous contentOuyang et al. [2022]. Reliability pertains to the model’sstability when confronted with unconventional data in long-tail scenarios, such as interference from adversarial exam-ples or with decision shortcuts Geirhos et al. [2020]. In thefollowing parts, we will delve into how different compres-sion methods impact these three crucial aspects of modelperformance.

# 7.1.1 Hallucination

The ability of an LLM to suppress hallucinations is crit-ically affected by modifications to its parameters. Accord-ing to previous research findings, factual knowledge is oftenstored within the Feed-Forward Networks (FFNs) of Trans-former modules. Therefore, when employing quantizationor structured compression methods, special attention shouldbe paid to the output calibration of the FFN layers. A viableapproach to addressing this concern involves identificationof key FFN layers, i.e., utilizing neuron-level interpretationtechniques from prior research Meng et al. [2022] to iden-tify the FFN layers that are crucial for storing knowledge.These layers are deemed important because they containweights and representations that are integral to the model’sability to recall and utilize factual information accurately.

For parts identified as crucial in storing knowledge, thequantization precision should be selectively enhanced. Thismeans that while the overall model undergoes quantizationto reduce its size and computational requirements, the quan-tization process for these critical FFN layers is adjusted tomaintain a higher level of precision. This selective approachhelps in preserving the integrity and accuracy of the storedfactual knowledge, thereby mitigating the risk of output hal-lucinations.

In the context of pruning, which involves removingweights or neurons deemed less important to streamline themodel, it is essential to retain the identified important FFNlayers. By preserving these layers, the model maintainsits core capability to recall and process factual knowledge,which is essential for ensuring the accuracy of its outputsand reducing the likelihood of generating hallucinated con-tent.

# 7.2. Safety Alignment

Based on previous research findings Yuan et al. [2023c],moderate model compression, such as 8-bit quantization,does not significantly compromise the safety capabilities

of models. However, it may render models more suscepti-ble to certain jailbreak attacks—a direction that prior stud-ies have seldom covered Deng et al. [2023]. Consequently,we recommend conducting comprehensive red teaming be-fore deploying these compressed models. Moreover, knowl-edge transfer-based approaches can substantially weakenthe safety of models. Therefore, we advise re-finetunesmaller models after completing knowledge transfer.

# 7.3. OOD Generalization

Large language models, when deployed in real-world sce-narios, are often influenced by decision shortcuts, leadingto erroneous decisions within the long-tail subgroup dis-tributions Geirhos et al. [2020]. As demonstrated by pre-vious research Yuan et al. [2023b], neural networks sub-jected to quantization compression exhibit significant per-formance disparities across different subgroups within thesame task, with errors in judgment frequently occurring inlong-tail subgroups that rely on decision shortcuts presentin the context. Furthermore, kv-cache compression, a com-monly used technique to enhance the inference efficiency oflarge language models in practice, relies on randomly dis-carding tokens in the attention matrix during inference. Thismethod further exacerbates the model’s reliance on decisionshortcuts. Therefore, it is advisable to consider integratingcorresponding robustness enhancement methods in down-stream specific scenarios, such as the invariant test-time op-timization techniques mentioned in prior studies Ma et al.[2024].

# 7.4. Efficient Large Multimodal Models

# 7.4.1 Large Multimodal Models (LMMs)

Large Multimodal Models (LMMs), particularly VisualLanguage Models (VLMs), have emerged as a promisingavenue for creating general-purpose assistants, showcasingsignificant enhancements in perception and reasoning ca-pabilities. These models leverage LLMs as their cogni-tive core, enriching multimodal (MM) tasks with robustlanguage generation, zero-shot transfer capabilities, andIn-Context Learning. Foundation models across differentmodalities provide high-quality representations. A criticalchallenge for LMMs is integrating LLMs with models fromother modalities to facilitate collaborative inference effec-tively. The primary focus has been on improving modal-ity alignment and aligning with human intentions through aMM Pre-Training + MM Instruction-Tuning pipeline. Twosurveys, [Yin et al., 2023b, Zhang et al., 2024a], delve intoLMMs in detail.

# 7.4.2 Efficient LMMs

The need for cross-modality capabilities in resource-limitedscenarios has become increasingly apparent. Despite

LMMs’ advancements, their large-scale training and de-ployment incur significant computational costs, necessi-tating efficient parallel device implementations. Google’sGemini [Team et al., 2023] leads in efficient LMMs, achiev-ing state-of-the-art performance on multimodal benchmarksand introducing mobile-scale LMMs suitable for low-memory devices. However, Gemini remains closed-source.Open-source initiatives, like LLaVA-v1.5, utilize advancedcompression techniques, such as 4/8 bit quantization via bit-sandbytes [Dettmers et al., 2022], for more on compressiontechniques, see Section 3.

Further efforts towards efficient LMMs include Mo-bileVLM [Chu et al., 2023], which develops compact LLMsand an efficient multimodal feature projector, and its succes-sor, MobileVLM-v2 [Chu et al., 2024], which explores im-proved training strategies for mobile scenarios. TinyGPT-V [Yuan et al., 2023a] leverages the advanced Phi-2 [Java-heripi et al., 2023] LLM to surpass the performance of sig-nificantly larger models. Similarly, LLaVA-Phi [Zhu et al.,2024] and Vary-toy [Wei et al., 2024] introduce smallerbackbones and enhanced vocabularies for broader general-izability. TinyLLaVA [Zhou et al., 2024] investigates theimpacts of architectural choices, data quality, and train-ing strategies, demonstrating that smaller LMMs can matchthe performance of their larger counterparts with optimizeddata and training. MoE-LLaVA [Lin et al., 2024] adaptsMixture of Experts (MoE) [Yuksel et al., 2012] to mitigatemodel degradation due to sparsity.

# 7.5. Long Context Modeling

When used for tasks like Chatbot or document summariza-tion tools, Large Language Models’ long context languagemodeling and reasoning capabilities are challenged. Themodel, however, are usually trained on the general pretrain-ing corpus that usually comprises text snippets, not longenough to serve as high-quality training examples for LLMto learn. To alleviate the insufficient long context capa-bilities of the pre-trained models, abundant works have at-tempted to attach the problem from different angles. Forthis section of the discussion, we mainly focus on alterna-tive attention mechanisms in 7.5.1, cache compression andcontext retrieval, and position encoding modifications. Formore detailed studies and a more comprehensive review ofthe issue of LLM long-context modeling, we refer interestedreaders to the recent survey specifically on this topic Huanget al. [2023b].

# 7.5.1 Alternative Attention Design

Lying in the core of transformer architecture is the self-attention mechanism. For decoder-only model inferencewith KV cache, if the past context is long, attending to allpast keys computing with past values incurs both compu-

tation and memory bottleneck. Prior works found that notall past tokens need to be attended to preserve the modelinference performance. Landmark Attention (Mohtashamiand Jaggi [2023]) introduced the special landmark atten-tion into the sequence to summarize the following blockof the tokens’ information. The new query would attendto the landmark tokens first to determine whether the fol-lowing tokens inside the block are needed for predicting thenext word, thus reducing the attention computation whilemaintaining the random-access nature of attention. Ear-lier, Funnel-Transformer (Dai et al. [2020]) also approachesthe same goal of conducting attention in the block level.Differently, they introduce a down-sampling during the en-coder portion and an up-sampling method during the de-coder portion. The reduced FLOPs allow them to builda deeper and wider model that outperforms the originalmodel with the same compute budget. On the other hand,Longformer (Beltagy et al. [2020]) gives an early attemptto combine sliding window attention and global attention,in which they only allow a few pre-defined tokens to at-tend to all tokens in the sequence, while other tokens dosliding window attention and also attend to these selectedglobal attending tokens. Concurrently, ETC (Ainslie et al.[2020]) introduces a 2-level hierarchy to the input so thatthe plain long input tokens can do sliding window atten-tion, while a global input of auxiliary tokens extracted anddownsampled from the original input can do normal atten-tion. The plain inputs are allowed to attend to the globalinputs, thus acquiring the global context information. Simi-larly, LongT5 (Guo et al. [2022b]) proposes an even simplermethod to directly downsampled the long and prior contextusing mean pooling with chunksize of 16, the downsam-pled keys and values are directly appended in front of therest of the input sequence. Then the rest of the model cando sliding window attention with the additional attention tothe downsampled context summary tokens at the front. Un-der the LLM era when pretraining becomes prohibitivelycostly, StreamingLLM (Xiao et al. [2023b]) and LM Infi-nite (Han et al. [2023]) concurrently propose a sink plussliding window attention pattern to LLM as a plug-in toolto boost the LLM’s long context ability. StreamingLLMin particular points out that due to the mechanism of soft-max operation in the transformer, the beginning tokens ofthe input are crucial to maintaining the self-attention per-formance. The modified attention mask is shown to achievestrong results in long-context language modeling withoutadditional model finetuning. Besides, $H _ { 2 } O$ (Zhang et al.[2024b]) reduces the self-attention computational complex-ity by only attending to tokens of interest out from the priorprefilled input context. To do that, they build an empiricaloracle for selecting tokens. The method is shown to benefitLLM in long-context regime.

# 7.5.2 Recurrence and Retrieval

Transformer-XL Dai et al. [2019] proposes to introducethe segment-level recurrence structure to the LanguageModel to boost the current language model in long-contextcapabilities. The method stores the last layer output ofone previous segment to append to the current layer,thus drastically increasing the dependency distance ofthe model. Segatron (Bai et al. [2021]) and CompressiveTransformer (Rae et al. [2019]) extend upon the previousidea. Segatron boosts the segment-level recurrence fromthe position embeddings on the token level, sentencelevel, and beyond through segment-aware mechanisms.Compressive Transformer proposes a second-level com-pressed memory FIFO queue so that the past segmentcontext should not be discarded but compressed by theircustomized function and stored in the queue to elongatethe length of context dependency. In the LLM era, Dy-namic Memory Compression Nawrot et al. [2024] alsofollows the recurrence idea with compressed context todynamically decide how to compress the previous contextinformation, thus reducing the sequence length of theattention while preserving the distant information. Besides,other than segment-level, Fan et al. [2020] studies theretrospective recurrence. Memorizing Transformer (Wuet al. [2022]) and Memformer (Wu et al. [2020]) com-bines retrieval and local cache with a forgetting mechanism.

On the other hand, following on trend that not every pasttoken is needed for the current token generation next, thepast KV cache can be placed physically further in distance,i.e. secondary memory, so that when only needed, the spe-cific key-value pair will be retrieved. Thus, another wayto boost the LLM is through Retrieval-Augmented Gener-ation (RAG). RAG is a heated topic in its own light andcontains abundant techniques and past works. Due to thelimited scope of our paper, we kindly refer to the followingcomprehensive surveys on RAG for interested readers: Gaoet al. [2023] and Zhao et al. [2024]. To address the particu-lar motivation underlined long-context capabilities broughtup at the beginning of this section, LangChain Pandya andHolia [2023] is the popular way to mitigate the Chatbot’spast long conversation through retrieval. LangChain is anopen-source tool that specializes in using LLM to computeembedding for user-input long documents and files, so thatlater according to the user’s prompt, top-relevant contentswill be retrieved through the cosine-similarity metrics. Be-sides, there are fast-growing other relevant works(Borgeaudet al. [2021], Bertsch et al. [2024], Zhong et al. [2023],Zhou et al. [2023b], Kynoch et al. [2023], Modarressi et al.[2023], Guu et al. [2020], Wang et al. [2024b]) under theretrieval for long-context settings.

# 7.5.3 Maneuvering Position Encodings

During pretraining, the position encoding of the trans-former hasn’t seen an input sequence length longer than afixed limit. Also, because the position encoding is usuallybased on a triangular function, the vanilla transformeris unable to extrapolate to unfamiliar longer sequencelengths. Earlier techniques add attention bias to theattention map before the softmax operation. ALiBi (Presset al. [2021]) introduces heuristics to design such attentionbias, achieving early success in long-context extrapolationtasks for transformer architecture. Besides, Kerple (Chiet al. [2022]) and Sandwich building on the prior works,introduce trainable parameters to construct the attentionbias matrix or construct attention bias through sinusoidalproperties of position encoding.

On the other hand, another lively line of work delves intoadjusting the RoPE. Inspired by the Neural Tangent Ker-nel (NTK) theory, NTK-aware Scaled RoPE (Xiong et al.[2023]) modifies the base parameter of the RoPE; LEX (Sunet al. [2022]) and PermuteFormer (Chen [2021]) adds anexponential decay term; Positional Interpolation places lin-ear scaling to every token; Dynamic-NTK (Huang and Yau[2019]) gradually increases the scaling ratio. In the LLMera, YaRN (Peng et al. [2023]) linearly scales query and keywith a temperature factor. Giraffe (Pal et al. [2023]) findsthat high-frequency terms are hurt by the low-frequencyterms that are usually ill-trained and proposes a scalingmechanism based on the power law to protect the well-trained high-frequency information.

# 8. Conclusion

In this work, we review on efficient large language model(LLM) inference. For this practice-driven topic, our com-prehensive study goes beyond conventional literature re-views by providing both an overview of existing researchand the development of a roofline model. Our first step isto develop a roofline model, which enables us to pinpointbottlenecks in LLM deployments, allowing researchers toresort to more specific deployment strategies. By meticu-lously assembling the latest developments in the field, oursurvey spans an array of pivotal areas, including innovationsin weight optimization techniques, enhancements in decod-ing algorithms, as well as advancements in hardware andsystem-level optimizations. It is important to note that thisproject will be updated and maintained.

# References



Abts, D., Kimmell, G., Ling, A., Kim, J., Boyd, M., Bitar, A.,Parmar, S., Ahmed, I., DiCecco, R., Han, D., et al. (2022).A software-defined tensor streaming multiprocessor for large-scale machine learning. In Proceedings of the 49th Annual In-ternational Symposium on Computer Architecture, pages 567–580. 23





Agarwal, R., Vieillard, N., Stanczyk, P., Ramos, S., Geist, M.,and Bachem, O. (2023). Gkd: Generalized knowledge dis-tillation for auto-regressive sequence models. arXiv preprintarXiv:2306.13649. 12





Agrawal, A., Panwar, A., Mohan, J., Kwatra, N., Gulavani, B. S.,and Ramjee, R. (2023). Sarathi: Efficient llm inference bypiggybacking decodes with chunked prefills. arXiv preprintarXiv:2308.16369. 22





Ainslie, J., Ontanon, S., Alberti, C., Cvicek, V., Fisher, Z., Pham,P., Ravula, A., Sanghai, S., Wang, Q., and Yang, L. (2020).Etc: Encoding long and structured inputs in transformers. arXivpreprint arXiv:2004.08483. 27





Alizadeh, K., Mirzadeh, I., Belenko, D., Khatamifard, K., Cho,M., Del Mundo, C. C., Rastegari, M., and Farajtabar, M. (2023).Llm in a flash: Efficient large language model inference withlimited memory. arXiv preprint arXiv:2312.11514. 21





Aminabadi, R. Y., Rajbhandari, S., Awan, A. A., Li, C., Li, D.,Zheng, E., Ruwase, O., Smith, S., Zhang, M., Rasley, J., et al.(2022). Deepspeed-inference: enabling efficient inference oftransformer models at unprecedented scale. In SC22: Interna-tional Conference for High Performance Computing, Network-ing, Storage and Analysis, pages 1–15. IEEE. 20, 21





Bae, S., Ko, J., Song, H., and Yun, S.-Y. (2023). Fast and ro-bust early-exiting framework for autoregressive language mod-els with synchronized parallel decoding. 13, 14, 15





Bai, H., Shi, P., Lin, J., Xie, Y., Tan, L., Xiong, K., Gao, W.,and Li, M. (2021). Segatron: Segment-aware transformer forlanguage modeling and understanding. In Proceedings of theAAAI Conference on Artificial Intelligence, volume 35, pages12526–12534. 27





Baier-Reinio, A. and Sterck, H. D. (2020). N-ode transformer: Adepth-adaptive variant of the transformer using neural ordinarydifferential equations. 13





Behdin, K., Acharya, A., Gupta, A., Keerthi, S., and Mazumder,R. (2023). Quantease: Optimization-based quantization forlanguage models–an efficient and intuitive algorithm. arXivpreprint arXiv:2309.01885. 8





Beltagy, I., Peters, M. E., and Cohan, A. (2020). Longformer: Thelong-document transformer. arXiv preprint arXiv:2004.05150.27





Bertsch, A., Alon, U., Neubig, G., and Gormley, M. (2024). Un-limiformer: Long-range transformers with unlimited length in-put. Advances in Neural Information Processing Systems, 36.27





Borgeaud, S. et al. (2021). Improving language models by retriev-ing from trillions of tokens. arxiv e-prints, art. arXiv preprintarXiv:2112.04426. 27





Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhari-wal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al.(2020). Language models are few-shot learners. Advances inneural information processing systems, 33:1877–1901. 1, 16





Cai, T., Li, Y., Geng, Z., Peng, H., Lee, J. D., Chen, D., andDao, T. (2024). Medusa: Simple llm inference accelera-tion framework with multiple decoding heads. arXiv preprintarXiv:2401.10774. 18, 19





Chae, H., Song, Y., Ong, K. T.-i., Kwon, T., Kim, M., Yu,Y., Lee, D., Kang, D., and Yeo, J. (2023). Dialogue chain-of-thought distillation for commonsense-aware conversationalagents. arXiv preprint arXiv:2310.09343. 12





Chee, J., Cai, Y., Kuleshov, V., and De Sa, C. (2023). Quip: 2-bitquantization of large language models with guarantees. Ad-vances in Neural Information Processing Systems. 8





Chen, C., Borgeaud, S., Irving, G., Lespiau, J.-B., Sifre,L., and Jumper, J. (2023a). Accelerating large languagemodel decoding with speculative sampling. arXiv preprintarXiv:2302.01318. 17





Chen, P. (2021). Permuteformer: Efficient relative position encod-ing for long sequences. 28





Chen, T., Moreau, T., Jiang, Z., Zheng, L., Yan, E., Shen, H.,Cowan, M., Wang, L., Hu, Y., Ceze, L., et al. (2018). Tvm: Anautomated end-to-end optimizing compiler for deep learning.In 13th USENIX Symposium on Operating Systems Design andImplementation (OSDI 18), pages 578–594. 20





Chen, Y., Pan, X., Li, Y., Ding, B., and Zhou, J. (2023b). Ee-llm:Large-scale training and inference of early-exit large languagemodels with 3d parallelism. 14, 15





Chi, T.-C., Fan, T.-H., Ramadge, P. J., and Rudnicky, A. (2022).Kerple: Kernelized relative positional embedding for length ex-trapolation. Advances in Neural Information Processing Sys-tems, 35:8386–8399. 28





Choi, J., Park, J., Kyung, K., Kim, N. S., and Ahn, J. H. (2023).Unleashing the potential of pim: Accelerating large batched in-ference of transformer-based generative models. IEEE Com-puter Architecture Letters. 23





Choi, J., Wang, Z., Venkataramani, S., Chuang, P. I.-J., Srinivasan,V., and Gopalakrishnan, K. (2018). Pact: Parameterized clip-ping activation for quantized neural networks. arXiv preprintarXiv:1805.06085. 8





Chowdhery, A., Narang, S., Devlin, J., Bosma, M., Mishra, G.,Roberts, A., Barham, P., Chung, H. W., Sutton, C., Gehrmann,S., Schuh, P., Shi, K., Tsvyashchenko, S., Maynez, J., Rao, A.,Barnes, P., Tay, Y., Shazeer, N., Prabhakaran, V., Reif, E., Du,N., Hutchinson, B., Pope, R., Bradbury, J., Austin, J., Isard,M., Gur-Ari, G., Yin, P., Duke, T., Levskaya, A., Ghemawat,





S., Dev, S., Michalewski, H., Garcia, X., Misra, V., Robinson,K., Fedus, L., Zhou, D., Ippolito, D., Luan, D., Lim, H., Zoph,B., Spiridonov, A., Sepassi, R., Dohan, D., Agrawal, S., Omer-nick, M., Dai, A. M., Pillai, T. S., Pellat, M., Lewkowycz, A.,Moreira, E., Child, R., Polozov, O., Lee, K., Zhou, Z., Wang,X., Saeta, B., Diaz, M., Firat, O., Catasta, M., Wei, J., Meier-Hellstern, K., Eck, D., Dean, J., Petrov, S., and Fiedel, N.(2022). Palm: Scaling language modeling with pathways. 16





Chu, X., Qiao, L., Lin, X., Xu, S., Yang, Y., Hu, Y., Wei, F., Zhang,X., Zhang, B., Wei, X., et al. (2023). Mobilevlm: A fast, repro-ducible and strong vision language assistant for mobile devices.arXiv preprint arXiv:2312.16886. 26





Chu, X., Qiao, L., Zhang, X., Xu, S., Wei, F., Yang, Y., Sun, X.,Hu, Y., Lin, X., Zhang, B., et al. (2024). Mobilevlm v2: Fasterand stronger baseline for vision language model. arXiv preprintarXiv:2402.03766. 26





Corro, L. D., Giorno, A. D., Agarwal, S., Yu, B., Awadallah, A.,and Mukherjee, S. (2023). Skipdecode: Autoregressive skipdecoding with batching and caching for efficient llm inference.15





Courbariaux, M., Bengio, Y., and David, J.-P. (2015). Binarycon-nect: Training deep neural networks with binary weights duringpropagations. Advances in neural information processing sys-tems, 28. 8





Dai, Z., Lai, G., Yang, Y., and Le, Q. (2020). Funnel-transformer:Filtering out sequential redundancy for efficient language pro-cessing. Advances in neural information processing systems,33:4271–4282. 27





Dai, Z., Yang, Z., Yang, Y., Carbonell, J., Le, Q. V., andSalakhutdinov, R. (2019). Transformer-xl: Attentive lan-guage models beyond a fixed-length context. arXiv preprintarXiv:1901.02860. 27





Dao, T. (2023). Flashattention-2: Faster attention with better par-allelism and work partitioning. 20





Dao, T., Fu, D., Ermon, S., Rudra, A., and Re, C. (2022). Flashat- ´tention: Fast and memory-efficient exact attention with io-awareness. Advances in Neural Information Processing Sys-tems, 35:16344–16359. 20





Dao, T., Haziza, D., Massa, F., and Sizov, G. (2023). Flash-decoding for long-context inference. 20





Darvish Rouhani, B., Zhao, R., Elango, V., Shafipour, R., Hall,M., Mesmakhosroshahi, M., More, A., Melnick, L., Golub, M.,Varatkar, G., et al. (2023). With shared microexponents, a littleshifting goes a long way. In Proceedings of the 50th AnnualInternational Symposium on Computer Architecture, pages 1–13. 25





de Jong, M., Zemlyanskiy, Y., Ainslie, J., FitzGerald, N., Sanghai,S., Sha, F., and Cohen, W. (2023). Fido: Fusion-in-decoderoptimized for stronger performance and faster inference. 13





Deng, G., Liu, Y., Li, Y., Wang, K., Zhang, Y., Li, Z., Wang, H.,Zhang, T., and Liu, Y. (2023). Jailbreaker: Automated jailbreakacross multiple large language model chatbots. arXiv preprintarXiv:2307.08715. 26





Dettmers, T., Lewis, M., Belkada, Y., and Zettlemoyer, L. (2022).Llm. int8 (): 8-bit matrix multiplication for transformers atscale. arXiv preprint arXiv:2208.07339. 8, 26





Dettmers, T., Pagnoni, A., Holtzman, A., and Zettlemoyer, L.(2023a). Qlora: Efficient finetuning of quantized llms. arXivpreprint arXiv:2305.14314. 10





Dettmers, T., Svirschevski, R., Egiazarian, V., Kuznedelev, D.,Frantar, E., Ashkboos, S., Borzunov, A., Hoefler, T., andAlistarh, D. (2023b). Spqr: A sparse-quantized representa-tion for near-lossless llm weight compression. arXiv preprintarXiv:2306.03078. 8





Dettmers, T. and Zettlemoyer, L. (2023). The case for 4-bit preci-sion: k-bit inference scaling laws. In International Conferenceon Machine Learning, pages 7750–7774. PMLR. 8





Devaux, F. (2019a). The true processing in memory accelerator. In2019 IEEE Hot Chips 31 Symposium (HCS), pages 1–24. IEEEComputer Society. 23





Devaux, F. (2019b). The true processing in memory accelerator. In2019 IEEE Hot Chips 31 Symposium (HCS), pages 1–24. IEEEComputer Society. 23





Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. (2019). Bert:Pre-training of deep bidirectional transformers for language un-derstanding. 16





Devvrit, Kudugunta, S., Kusupati, A., Dettmers, T., Chen, K.,Dhillon, I., Tsvetkov, Y., Hajishirzi, H., Kakade, S., Farhadi,A., and Jain, P. (2023). Matformer: Nested transformer forelastic inference. 15





Din, A. Y., Karidi, T., Choshen, L., and Geva, M. (2023). Jumpto conclusions: Short-cutting transformers with linear transfor-mations. 14





Ding, T., Chen, T., Zhu, H., Jiang, J., Zhong, Y., Zhou, J., Wang,G., Zhu, Z., Zharkov, I., and Liang, L. (2023). The efficiencyspectrum of large language models: An algorithmic survey.arXiv preprint arXiv:2312.00678. 3, 14





Dong, Z., Yao, Z., Gholami, A., Mahoney, M. W., and Keutzer,K. (2019). Hawq: Hessian aware quantization of neural net-works with mixed-precision. In Proceedings of the IEEE/CVFInternational Conference on Computer Vision, pages 293–302.8





Du, N., Huang, Y., Dai, A. M., Tong, S., Lepikhin, D., Xu, Y.,Krikun, M., Zhou, Y., Yu, A. W., Firat, O., et al. (2022). Glam:Efficient scaling of language models with mixture-of-experts.In International Conference on Machine Learning, pages 5547–5569. PMLR. 16





Elbayad, M., Gu, J., Grave, E., and Auli, M. (2020). Depth-adaptive transformer. 13, 14, 15





Fan, A., Lavril, T., Grave, E., Joulin, A., and Sukhbaatar, S.(2020). Addressing some limitations of transformers with feed-back memory. arXiv preprint arXiv:2002.09402. 27





Fedus, W., Dean, J., and Zoph, B. (2022). A review of sparse ex-pert models in deep learning. arXiv preprint arXiv:2209.01667.16





Fedus, W., Zoph, B., and Shazeer, N. (2021). Switch transformers:Scaling to trillion parameter models with simple and efficientsparsity.(2021). arXiv preprint cs.LG/2101.03961. 15





Frantar, E. and Alistarh, D. (2023). Sparsegpt: Massive languagemodels can be accurately pruned in one-shot. ICML. 11





Frantar, E., Ashkboos, S., Hoefler, T., and Alistarh, D. (2022).Gptq: Accurate post-training quantization for generative pre-trained transformers. arXiv preprint arXiv:2210.17323. 8





Fu, Y., Bailis, P., Stoica, I., and Zhang, H. (2023a). Breaking thesequential dependency of llm inference using lookahead decod-ing. 19





Fu, Y., Peng, H., Ou, L., Sabharwal, A., and Khot, T. (2023b).Specializing smaller language models towards multi-step rea-soning. arXiv preprint arXiv:2301.12726. 12





Gao, Y., Xiong, Y., Gao, X., Jia, K., Pan, J., Bi, Y., Dai, Y., Sun, J.,and Wang, H. (2023). Retrieval-augmented generation for largelanguage models: A survey. arXiv preprint arXiv:2312.10997.27





Geirhos, R., Jacobsen, J.-H., Michaelis, C., Zemel, R., Brendel,W., Bethge, M., and Wichmann, F. A. (2020). Shortcut learn-ing in deep neural networks. Nature Machine Intelligence,2(11):665–673. 25, 26





Geva, M., Caciularu, A., Wang, K. R., and Goldberg, Y. (2022).Transformer feed-forward layers build predictions by promot-ing concepts in the vocabulary space. 13





Ghazvininejad, M., Levy, O., Liu, Y., and Zettlemoyer, L. (2019).Mask-predict: Parallel decoding of conditional masked lan-guage models. 19





Gholami, A., Kim, S., Dong, Z., Yao, Z., Mahoney, M. W., andKeutzer, K. (2022). A survey of quantization methods for effi-cient neural network inference. In Low-Power Computer Vision,pages 291–326. Chapman and Hall/CRC. 6, 24





Gou, J., Yu, B., Maybank, S. J., and Tao, D. (2021). Knowledgedistillation: A survey. International Journal of Computer Vi-sion, 129:1789–1819. 11





Graphcore (2024). Tasks and tutorials using graphcore’s ipuwith hugging face. https://github.com/graphcore/Gradient-HuggingFace. Accessed on March 10, 2024.23





Groq (2023). Groq sets new large language model perfor-mance record of 300 tokens per second per user on metaai foundational llm llama 2.70b. https://wow.groq.com/groq- sets- new- large- language- model-performance - record - of - 300 - tokens - per -second- per- user- on- meta- ai- foundational-llm-llama-2-70b/. 23





Gu, J., Bradbury, J., Xiong, C., Li, V. O., and Socher, R. (2017).Non-autoregressive neural machine translation. arXiv preprintarXiv:1711.02281. 19





Gu, J. and Kong, X. (2020). Fully non-autoregressive neural ma-chine translation: Tricks of the trade. 19





Gu, Y., Dong, L., Wei, F., and Huang, M. (2023). Knowl-edge distillation of large language models. arXiv preprintarXiv:2306.08543. 11





Guo, C., Tang, J., Hu, W., Leng, J., Zhang, C., Yang, F., Liu, Y.,Guo, M., and Zhu, Y. (2023a). Olive: Accelerating large lan-guage models via hardware-friendly outlier-victim pair quanti-zation. In Proceedings of the 50th Annual International Sym-posium on Computer Architecture, pages 1–15. 9, 24





Guo, C., Zhang, C., Leng, J., Liu, Z., Yang, F., Liu, Y., Guo,M., and Zhu, Y. (2022a). Ant: Exploiting adaptive numeri-cal data type for low-bit deep neural network quantization. In2022 55th IEEE/ACM International Symposium on Microarchi-tecture (MICRO), pages 1414–1433. IEEE. 24





Guo, H., Greengard, P., Xing, E. P., and Kim, Y. (2023b). Lq-lora:Low-rank plus quantized matrix decomposition for efficientlanguage model finetuning. arXiv preprint arXiv:2311.12023.10





Guo, J., Xu, L., and Chen, E. (2020). Jointly masked sequence-to-sequence model for non-autoregressive neural machine transla-tion. In Jurafsky, D., Chai, J., Schluter, N., and Tetreault, J.,editors, Proceedings of the 58th Annual Meeting of the Asso-ciation for Computational Linguistics, pages 376–385, Online.Association for Computational Linguistics. 19





Guo, M., Ainslie, J., Uthus, D., Ontanon, S., Ni, J., Sung, Y.-H.,and Yang, Y. (2022b). Longt5: Efficient text-to-text transformerfor long sequences. 27





Guo, S., Xu, J., Zhang, L. L., and Yang, M. (2023c). Compresso:Structured pruning with collaborative prompting learns com-pact large language models. arXiv preprint arXiv:2310.05015.11





Guu, K., Lee, K., Tung, Z., Pasupat, P., and Chang, M. (2020).Retrieval augmented language model pre-training. In Inter-national conference on machine learning, pages 3929–3938.PMLR. 27





Han, C., Wang, Q., Xiong, W., Chen, Y., Ji, H., and Wang, S.(2023). Lm-infinite: Simple on-the-fly length generalizationfor large language models. arXiv preprint arXiv:2308.16137.27





He, Z., Zhong, Z., Cai, T., Lee, J. D., and He, D. (2023).Rest: Retrieval-based speculative decoding. arXiv preprintarXiv:2311.08252. 18





Hinton, G., Vinyals, O., and Dean, J. (2015). Distilling the knowl-edge in a neural network. arXiv preprint arXiv:1503.02531. 11





Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., Cai,T., Rutherford, E., de Las Casas, D., Hendricks, L. A., Welbl,J., Clark, A., Hennigan, T., Noland, E., Millican, K., van denDriessche, G., Damoc, B., Guy, A., Osindero, S., Simonyan, K.,Elsen, E., Rae, J. W., Vinyals, O., and Sifre, L. (2022). Trainingcompute-optimal large language models. 15





Holmes, C., Tanaka, M., Wyatt, M., Awan, A. A., Rasley,J., Rajbhandari, S., Aminabadi, R. Y., Qin, H., Bakhtiari,A., Kurilenko, L., et al. (2024). Deepspeed-fastgen: High-throughput text generation for llms via mii and deepspeed-inference. arXiv preprint arXiv:2401.08671. 22





Hong, K., Dai, G., Xu, J., Mao, Q., Li, X., Liu, J., Chen, K.,Dong, H., and Wang, Y. (2023). Flashdecoding++: Fasterlarge language model inference on gpus. arXiv preprintarXiv:2311.01282. 21





Hong, S., Moon, S., Kim, J., Lee, S., Kim, M., Lee, D., andKim, J.-Y. (2022). Dfx: A low-latency multi-fpga appliancefor accelerating transformer-based text generation. In 202255th IEEE/ACM International Symposium on Microarchitec-ture (MICRO), pages 616–630. IEEE. 25





Hooper, C., Kim, S., Mohammadzadeh, H., Mahoney, M. W.,Shao, Y. S., Keutzer, K., and Gholami, A. (2024). Kvquant:Towards 10 million context length llm inference with kv cachequantization. arXiv preprint arXiv:2401.18079. 9





Hou, L., Huang, Z., Shang, L., Jiang, X., Chen, X., and Liu, Q.(2020). Dynabert: Dynamic bert with adaptive width and depth.13





Hsieh, C.-Y., Li, C.-L., Yeh, C.-K., Nakhost, H., Fujii, Y., Rat-ner, A., Krishna, R., Lee, C.-Y., and Pfister, T. (2023). Distill-ing step-by-step! outperforming larger language models withless training data and smaller model sizes. arXiv preprintarXiv:2305.02301. 12





Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S.,Wang, L., and Chen, W. (2021). Lora: Low-rank adaptation oflarge language models. arXiv preprint arXiv:2106.09685. 10





Hu, Z., Lan, Y., Wang, L., Xu, W., Lim, E.-P., Lee, R. K.-W., Bing,L., and Poria, S. (2023). Llm-adapters: An adapter family forparameter-efficient fine-tuning of large language models. arXivpreprint arXiv:2304.01933. 10





Huang, J. and Yau, H.-T. (2019). Dynamics of deep neural net-works and neural tangent hierarchy. 28





Huang, L., Yu, W., Ma, W., Zhong, W., Feng, Z., Wang, H.,Chen, Q., Peng, W., Feng, X., Qin, B., et al. (2023a). Asurvey on hallucination in large language models: Principles,taxonomy, challenges, and open questions. arXiv preprintarXiv:2311.05232. 25





Huang, W., Liu, Y., Qin, H., Li, Y., Zhang, S., Liu, X., Magno,M., and Qi, X. (2024). Billm: Pushing the limit of post-trainingquantization for llms. arXiv preprint arXiv:2402.04291. 9





Huang, Y., Chen, Y., Yu, Z., and McKeown, K. (2022). In-contextlearning distillation: Transferring few-shot learning ability ofpre-trained language models. arXiv preprint arXiv:2212.10670.12





Huang, Y., Xu, J., Jiang, Z., Lai, J., Li, Z., Yao, Y., Chen, T.,Yang, L., Xin, Z., and Ma, X. (2023b). Advancing transformerarchitecture in long-context large language models: A compre-hensive survey. arXiv preprint arXiv:2311.12351. 26





Hubara, I., Courbariaux, M., Soudry, D., El-Yaniv, R., and Ben-gio, Y. (2016). Binarized neural networks. Advances in neuralinformation processing systems, 29. 9





HuggingFace (2022). Hugging face accelerate. 21





Javaheripi, M., Bubeck, S., Abdin, M., Aneja, J., Bubeck, S.,Mendes, C. C. T., Chen, W., Del Giorno, A., Eldan, R., Gopi,S., et al. (2023). Phi-2: The surprising power of small languagemodels. Microsoft Research Blog. 26





Jawahar, G., Mukherjee, S., Liu, X., Kim, Y. J., Mageed, M. A.,Laks Lakshmanan, V., Hassan, A., Bubeck, S., and Gao, J.(2023). Automoe: Heterogeneous mixture-of-experts withadaptive computation for efficient neural machine translation.In Findings of the Association for Computational Linguistics:ACL 2023, pages 9116–9132. 16





Ji, Y., Wang, J., Li, J., Chen, Q., Chen, W., and Zhang, M.(2023). Early exit with disentangled representation and equian-gular tight frame. In Rogers, A., Boyd-Graber, J., and Okazaki,N., editors, Findings of the Association for Computational Lin-guistics: ACL 2023, pages 14128–14142, Toronto, Canada. As-sociation for Computational Linguistics. 15





Jiang, A. Q., Sablayrolles, A., Roux, A., Mensch, A., Savary, B.,Bamford, C., Chaplot, D. S., Casas, D. d. l., Hanna, E. B.,Bressand, F., et al. (2024). Mixtral of experts. arXiv preprintarXiv:2401.04088. 16





Jiang, Y., Chan, C., Chen, M., and Wang, W. (2023). Lion: Adver-sarial distillation of closed-source large language model. arXivpreprint arXiv:2305.12870. 12





Jonatan, G., Cho, H., Son, H., Wu, X., Livesay, N., Mora, E.,Shivdikar, K., Abellan, J. L., Joshi, A., Kaeli, D., et al. (2024).´Scalability limitations of processing-in-memory using real sys-tem evaluations. Proceedings of the ACM on Measurement andAnalysis of Computing Systems, 8(1):1–28. 23





Kao, S.-C., Subramanian, S., Agrawal, G., Yazdanbakhsh, A., andKrishna, T. (2023). Flat: An optimized dataflow for mitigatingattention bottlenecks. In Proceedings of the 28th ACM Interna-tional Conference on Architectural Support for ProgrammingLanguages and Operating Systems, Volume 2, pages 295–310.25





Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess,B., Child, R., Gray, S., Radford, A., Wu, J., and Amodei, D.(2020). Scaling laws for neural language models. 15





Kim, J., Lee, J. H., Kim, S., Park, J., Yoo, K. M., Kwon, S. J., andLee, D. (2023a). Memory-efficient fine-tuning of compressedlarge language models via sub-4-bit integer quantization. arXivpreprint arXiv:2305.14152. 10





Kim, M., Lee, S., Lee, J., Hong, S., Chang, D.-S., Sung,W., and Choi, J. (2023b). Token-scaled logit distillation forternary weight generative language models. arXiv preprintarXiv:2308.06744. 8





Kim, S., Hooper, C., Gholami, A., Dong, Z., Li, X., Shen, S.,Mahoney, M. W., and Keutzer, K. (2023c). Squeezellm: Dense-and-sparse quantization. arXiv preprint arXiv:2306.07629. 8





Kim, S., Hooper, C., Wattanawong, T., Kang, M., Yan, R., Genc,H., Dinh, G., Huang, Q., Keutzer, K., Mahoney, M. W., Shao,Y. S., and Gholami, A. (2023d). Full stack optimization oftransformer inference: a survey. 13





Kim, S., Mangalam, K., Moon, S., Malik, J., Mahoney, M. W.,Gholami, A., and Keutzer, K. (2023e). Speculative decodingwith big little decoder. 17





Kishore Kumar, N. and Schneider, J. (2017). Literature survey onlow rank approximation of matrices. Linear and MultilinearAlgebra, 65(11):2212–2244. 12





Kossmann, F., Jia, Z., and Aiken, A. (2022). Optimizing mix-ture of experts using dynamic recompilations. arXiv preprintarXiv:2205.01848. 16





Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H.,Gonzalez, J., Zhang, H., and Stoica, I. (2023). Efficient mem-ory management for large language model serving with page-dattention. In Proceedings of the 29th Symposium on OperatingSystems Principles, pages 611–626. 21





Kwon, Y., Lee, S. H., Lee, J., Kwon, S., Ryu, J., Son, J., O, S.,Yu, H., Lee, H., Kim, S. Y., Cho, Y., Kim, J. G., Choi, J., Shin,H., Kim, J., Phuah, B., Kim, H., Song, M. J., Choi, A., Kim,D., Kim, S., Kim, E., Wang, D., Kang, S., Ro, Y., Seo, S.,Song, J., Youn, J., Sohn, K., and Kim, N. S. (2021). 25.4 A20nm 6gb function-in-memory dram, based on HBM2 with a1.2tflops programmable computing unit using bank-level par-allelism, for machine learning applications. In IEEE Interna-tional Solid-State Circuits Conference, ISSCC 2021, San Fran-cisco, CA, USA, February 13-22, 2021, pages 350–352. IEEE.23





Kwon, Y., Vladimir, K., Kim, N., Shin, W., Won, J., Lee, M., Joo,H., Choi, H., Kim, G., An, B., et al. (2022). System architectureand software stack for gddr6-aim. In 2022 IEEE Hot Chips 34Symposium (HCS), pages 1–25. IEEE. 23





Kynoch, B., Latapie, H., and van der Sluis, D. (2023). Recallm:An adaptable memory mechanism with temporal understandingfor large language models. 27





Lan, T., Cai, D., Wang, Y., Huang, H., and Mao, X.-L. (2023).Copy is all you need. 18





Langroudi, H. F., Karia, V., Carmichael, Z., Zyarah, A., Pandit, T.,Gustafson, J. L., and Kudithipudi, D. (2021). Alps: Adaptivequantization of deep neural networks with generalized posits. InProceedings of the IEEE/CVF Conference on Computer Visionand Pattern Recognition, pages 3100–3109. 24





LeCun, Y., Denker, J., and Solla, S. (1989). Optimal brain damage.Advances in neural information processing systems, 2. 11





Lee, C., Jin, J., Kim, T., Kim, H., and Park, E. (2023). Owq:Lessons learned from activation outliers for weight quantizationin large language models. arXiv preprint arXiv:2306.02272. 8





Lee, J., Mansimov, E., and Cho, K. (2018). Deterministic non-autoregressive neural sequence modeling by iterative refine-ment. 19





Lee-Thorp, J. and Ainslie, J. (2022). Sparse mixers: Combiningmoe and mixing to build a more efficient bert. arXiv preprintarXiv:2205.12399. 16





Lefaudeux, B., Massa, F., Liskovich, D., Xiong, W., Caggiano, V.,Naren, S., Xu, M., Hu, J., Tintore, M., Zhang, S., Labatut, P.,Haziza, D., Wehrstedt, L., Reizenstein, J., and Sizov, G. (2022).xformers: A modular and hackable transformer modelling li-brary. https://github.com/facebookresearch/xformers. 20





Lepikhin, D., Lee, H., Xu, Y., Chen, D., Firat, O., Huang, Y.,Krikun, M., Shazeer, N., and Chen, Z. (2020). Gshard: Scal-ing giant models with conditional computation and automaticsharding. arXiv preprint arXiv:2006.16668. 15





Leviathan, Y., Kalman, M., and Matias, Y. (2023). Fast inferencefrom transformers via speculative decoding. 17





Li, L., Li, Q., Zhang, B., and Chu, X. (2023a). Norm tweaking:High-performance low-bit quantization of large language mod-els. arXiv preprint arXiv:2309.02784. 9





Li, L., Lin, Y., Chen, D., Ren, S., Li, P., Zhou, J., and Sun, X.(2021). Cascadebert: Accelerating inference of pre-trained lan-guage models via calibrated complete models cascade. 13





Li, L. H., Hessel, J., Yu, Y., Ren, X., Chang, K.-W., andChoi, Y. (2023b). Symbolic chain-of-thought distillation:Small models can also” think” step-by-step. arXiv preprintarXiv:2306.14050. 12





Li, Q., Zhang, Y., Li, L., Yao, P., Zhang, B., Chu, X., Sun,Y., Du, L., and Xie, Y. (2023c). Fptq: Fine-grained post-training quantization for large language models. arXiv preprintarXiv:2308.15987. 9





Li, Y., Wei, F., Zhang, C., and Zhang, H. (2024). Eagle: Spec-ulative sampling requires rethinking feature uncertainty. arXivpreprint arXiv:2401.15077. 18





Li, Y., Yu, Y., Liang, C., He, P., Karampatziakis, N., Chen, W., andZhao, T. (2023d). Loftq: Lora-fine-tuning-aware quantizationfor large language models. arXiv preprint arXiv:2310.08659.10





Li, Z., Lin, Z., He, D., Tian, F., Qin, T., Wang, L., and Liu, T.-Y. (2019). Hint-based training for non-autoregressive machinetranslation. arXiv preprint arXiv:1909.06708. 19





Li, Z., Liu, X., Zhu, B., Dong, Z., Gu, Q., and Keutzer, K. (2023e).Qft: Quantized full-parameter tuning of llms with affordableresources. arXiv preprint arXiv:2310.07147. 10





Liang, C., Jiang, H., Li, Z., Tang, X., Yin, B., and Zhao, T.(2023a). Homodistil: Homotopic task-agnostic distillation ofpre-trained transformers. arXiv preprint arXiv:2302.09632. 12





Liang, C., Zuo, S., Zhang, Q., He, P., Chen, W., and Zhao, T.(2023b). Less is more: Task-aware layer-wise distillation forlanguage model compression. In International Conference onMachine Learning, pages 20852–20867. PMLR. 12





Liang, T., Glossner, J., Wang, L., Shi, S., and Zhang, X. (2021).Pruning and quantization for deep neural network acceleration:A survey. Neurocomputing, 461:370–403. 11





Lin, B., Tang, Z., Ye, Y., Cui, J., Zhu, B., Jin, P., Zhang,J., Ning, M., and Yuan, L. (2024). Moe-llava: Mixtureof experts for large vision-language models. arXiv preprintarXiv:2401.15947. 26





Lin, J., Tang, J., Tang, H., Yang, S., Dang, X., and Han, S. (2023).Awq: Activation-aware weight quantization for llm compres-sion and acceleration. arXiv preprint arXiv:2306.00978. 8





Liu, M., Zeng, A., Wang, B., Zhang, P., Tang, J., and Dong, Y.(2024a). Apar: Llms can do auto-parallel auto-regressive de-coding. 19





Liu, W., Zhou, P., Zhao, Z., Wang, Z., Deng, H., and Ju, Q. (2020).Fastbert: a self-distilling bert with adaptive inference time. 13





Liu, X., Hu, L., Bailis, P., Stoica, I., Deng, Z., Cheung, A., andZhang, H. (2023a). Online speculative decoding. arXiv preprintarXiv:2310.07177. 18





Liu, X., Sun, T., He, J., Wu, J., Wu, L., Zhang, X., Jiang, H., Cao,Z., Huang, X., and Qiu, X. (2022). Towards efficient NLP:A standard evaluation and a strong baseline. In Carpuat, M.,de Marneffe, M.-C., and Meza Ruiz, I. V., editors, Proceedingsof the 2022 Conference of the North American Chapter of theAssociation for Computational Linguistics: Human LanguageTechnologies, pages 3288–3303, Seattle, United States. Associ-ation for Computational Linguistics. 13





Liu, Z., Oguz, B., Zhao, C., Chang, E., Stock, P., Mehdad, Y.,Shi, Y., Krishnamoorthi, R., and Chandra, V. (2023b). Llm-qat: Data-free quantization aware training for large languagemodels. arXiv preprint arXiv:2305.17888. 8, 24





Liu, Z., Wang, J., Dao, T., Zhou, T., Yuan, B., Song, Z., Shrivas-tava, A., Zhang, C., Tian, Y., Re, C., and Chen, B. (2023c). Dejavu: Contextual sparsity for efficient llms at inference time. 15





Liu, Z., Yuan, J., Jin, H., Zhong, S., Xu, Z., Braverman, V., Chen,B., and Hu, X. (2024b). Kivi: A tuning-free asymmetric 2bitquantization for kv cache. arXiv preprint arXiv:2402.02750. 9





Ma, H., Zhu, Y., Zhang, C., Zhao, P., Wu, B., Huang, L.-K., Hu, Q., and Wu, B. (2024). Invariant test-time adapta-tion for vision-language model generalization. arXiv preprintarXiv:2403.00376. 26





Ma, X., Fang, G., and Wang, X. (2023). Llm-pruner: On thestructural pruning of large language models. arXiv preprintarXiv:2305.11627. 11





Meng, K., Bau, D., Andonian, A., and Belinkov, Y. (2022). Locat-ing and editing factual associations in GPT. In Koyejo, S., Mo-hamed, S., Agarwal, A., Belgrave, D., Cho, K., and Oh, A., ed-itors, Advances in Neural Information Processing Systems 35:Annual Conference on Neural Information Processing Systems2022, NeurIPS 2022, New Orleans, LA, USA, November 28 -December 9, 2022. 25





Miao, X., Oliaro, G., Zhang, Z., Cheng, X., Jin, H., Chen, T.,and Jia, Z. (2023a). Towards efficient generative large languagemodel serving: A survey from algorithms to systems. arXivpreprint arXiv:2312.15234. 2





Miao, X., Oliaro, G., Zhang, Z., Cheng, X., Wang, Z., Wong,R. Y. Y., Chen, Z., Arfeen, D., Abhyankar, R., and Jia, Z.(2023b). Specinfer: Accelerating generative llm serving withspeculative inference and token tree verification. arXiv preprintarXiv:2305.09781. 17





Micikevicius, P., Stosic, D., Burgess, N., Cornea, M., Dubey, P.,Grisenthwaite, R., Ha, S., Heinecke, A., Judd, P., Kamalu, J.,et al. (2022). Fp8 formats for deep learning. arXiv preprintarXiv:2209.05433. 24





Modarressi, A., Imani, A., Fayyaz, M., and Schutze, H. (2023).¨Ret-llm: Towards a general read-write memory for large lan-guage models. 27





ModelTC (2024). Lightllm: A python-based llm inference andserving framework. https://github.com/ModelTC/lightllm. 22





Mohtashami, A. and Jaggi, M. (2023). Landmark attention:Random-access infinite context length for transformers. arXivpreprint arXiv:2305.16300. 27





Monea, G., Joulin, A., and Grave, E. (2023). Pass: Parallel specu-lative sampling. arXiv preprint arXiv:2311.13581. 18





Nawrot, P., Łancucki, A., Chochowski, M., Tarjan, D., and Ponti,´E. M. (2024). Dynamic memory compression: Retrofitting llmsfor accelerated inference. arXiv preprint arXiv:2403.09636. 27





Ning, X., Lin, Z., Zhou, Z., Wang, Z., Yang, H., and Wang, Y.(2023). Skeleton-of-thought: Large language models can doparallel decoding. 19





NVIDIA (2022). Inside the nvidia hopper architecture.https://www.nvidia.com/en-us/data-center/technologies/hopper- architecture/. Accessedon March 11, 2024. 24





Ortega, J. M. and Rheinboldt, W. C. (2000). Iterative solution ofnonlinear equations in several variables. SIAM. 19





Oseledets, I. V. (2011). Tensor-train decomposition. SIAM Journalon Scientific Computing, 33(5):2295–2317. 13





Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C.,Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al.(2022). Training language models to follow instructions withhuman feedback. Advances in Neural Information ProcessingSystems, 35:27730–27744. 1, 25





Pal, A., Karkhanis, D., Roberts, M., Dooley, S., Sundararajan, A.,and Naidu, S. (2023). Giraffe: Adventures in expanding contextlengths in llms. 28





Pandya, K. and Holia, M. (2023). Automating customer serviceusing langchain: Building custom open-source gpt chatbot fororganizations. arXiv preprint arXiv:2310.05421. 27





Park, E. et al. (2018). Energy-efficient neural network acceleratorbased on outlier-aware low-precision computation. In ISCA,pages 688–698. IEEE. 24





Park, G., Park, B., Kim, M., Lee, S., Kim, J., Kwon, B., Kwon,S. J., Kim, B., Lee, Y., and Lee, D. (2023). Lut-gemm: Quan-tized matrix multiplication based on luts for efficient infer-ence in large-scale generative language models. arXiv preprintarXiv:2206.09557. 8





Peng, B., Quesnelle, J., Fan, H., and Shippole, E. (2023). Yarn:Efficient context window extension of large language models.28





Press, O., Smith, N. A., and Lewis, M. (2021). Train short, testlong: Attention with linear biases enables input length extrapo-lation. arXiv preprint arXiv:2108.12409. 28





Qin, Y., Wang, Y., Deng, D., Zhao, Z., Yang, X., Liu, L., Wei, S.,Hu, Y., and Yin, S. (2023). Fact: Ffn-attention co-optimizedtransformer architecture with eager correlation prediction. InProceedings of the 50th Annual International Symposium onComputer Architecture, pages 1–14. 25





Rae, J. W., Potapenko, A., Jayakumar, S. M., and Lillicrap, T. P.(2019). Compressive transformers for long-range sequencemodelling. 27





Rajbhandari, S., Li, C., Yao, Z., Zhang, M., Aminabadi, R. Y.,Awan, A. A., Rasley, J., and He, Y. (2022). Deepspeed-moe:Advancing mixture-of-experts inference and training to powernext-generation ai scale. In International Conference on Ma-chine Learning, pages 18332–18346. PMLR. 16





Rotem, D., Hassid, M., Mamou, J., and Schwartz, R. (2023). Find-ing the sweet spot: Analysis and improvement of adaptive in-ference in low resource settings. 15





Sahu, G., Vechtomova, O., Bahdanau, D., and Laradji, I. H.(2023). Promptmix: A class boundary augmentationmethod for large language model distillation. arXiv preprintarXiv:2310.14192. 12





Santilli, A., Severino, S., Postolache, E., Maiorca, V., Mancusi,M., Marin, R., and Rodola, E. (2023). Accelerating transformer `inference for translation via parallel decoding. arXiv preprintarXiv:2305.10427. 19





Savinov, N., Chung, J., Binkowski, M., Elsen, E., and Oord, A.v. d. (2021). Step-unrolled denoising autoencoders for text gen-eration. arXiv preprint arXiv:2112.06749. 19





Scao, T. L., Fan, A., Akiki, C., Pavlick, E., Ilic, S., Hesslow, D.,´Castagne, R., Luccioni, A. S., Yvon, F., Gall ´ e, M., et al. (2022).´Bloom: A 176b-parameter open-access multilingual languagemodel. arXiv preprint arXiv:2211.05100. 1





Schotthofer, S., Zangrando, E., Kusch, J., Ceruti, G., and Tudisco, ¨F. (2022). Low-rank lottery tickets: finding efficient low-rankneural networks via matrix differential equations. Advances inNeural Information Processing Systems, 35:20051–20063. 12





Schuster, T., Fisch, A., Gupta, J., Dehghani, M., Bahri, D., Tran,V. Q., Tay, Y., and Metzler, D. (2022). Confident adaptive lan-guage modeling. 13, 14, 15





Schuster, T., Fisch, A., Jaakkola, T., and Barzilay, R. (2021). Con-sistent accelerated inference via confident adaptive transform-ers. 13





Schwartz, R., Stanovsky, G., Swayamdipta, S., Dodge, J., andSmith, N. A. (2020). The right tool for the job: Matchingmodel and instance complexities. In Jurafsky, D., Chai, J.,Schluter, N., and Tetreault, J., editors, Proceedings of the 58thAnnual Meeting of the Association for Computational Linguis-tics, pages 6640–6651, Online. Association for ComputationalLinguistics. 13





Shang, Y., Duan, B., Zong, Z., Nie, L., and Yan, Y. (2021). Lip-schitz continuity guided knowledge distillation. In Proceed-ings of the IEEE/CVF International Conference on ComputerVision, pages 10675–10684. 11





Shang, Y., Yuan, Z., and Dong, Z. (2024). Pb-llm: Partially bina-rized large language models. In ICLR. 8, 9





Shao, C., Zhang, J., Feng, Y., Meng, F., and Zhou, J. (2020). Min-imizing the bag-of-ngrams difference for non-autoregressiveneural machine translation. In Proceedings of the AAAI con-ference on artificial intelligence, volume 34, pages 198–205.19





Sharma, P., Ash, J. T., and Misra, D. (2023). The truth is in there:Improving reasoning in language models with layer-selectiverank reduction. arXiv preprint arXiv:2312.13558. 12





Sheng, Y., Zheng, L., Yuan, B., Li, Z., Ryabinin, M., Chen, B.,Liang, P., Re, C., Stoica, I., and Zhang, C. (2023). Flexgen:´High-throughput generative inference of large language models





with a single GPU. In Krause, A., Brunskill, E., Cho, K., Engel-hardt, B., Sabato, S., and Scarlett, J., editors, International Con-ference on Machine Learning, ICML 2023, 23-29 July 2023,Honolulu, Hawaii, USA, volume 202 of Proceedings of Ma-chine Learning Research, pages 31094–31116. PMLR. 21





Simoulin, A. and Crabbe, B. (2021). How many layers and why? ´An analysis of the model depth in transformers. In Kabbara, J.,Lin, H., Paullada, A., and Vamvas, J., editors, Proceedings ofthe 59th Annual Meeting of the Association for ComputationalLinguistics and the 11th International Joint Conference on Nat-ural Language Processing: Student Research Workshop, pages221–228, Online. Association for Computational Linguistics.13





Song, Y., Meng, C., Liao, R., and Ermon, S. (2021). Acceler-ating feedforward computation via parallel nonlinear equationsolving. In Meila, M. and Zhang, T., editors, Proceedings ofthe 38th International Conference on Machine Learning, vol-ume 139 of Proceedings of Machine Learning Research, pages9791–9800. PMLR. 19





Song, Y., Mi, Z., Xie, H., and Chen, H. (2023). Powerinfer: Fastlarge language model serving with a consumer-grade gpu. 15





Stern, M., Shazeer, N., and Uszkoreit, J. (2018). Blockwise paral-lel decoding for deep autoregressive models. Advances in Neu-ral Information Processing Systems, 31. 18





Stickland, A. C. and Murray, I. (2019). Bert and pals: Projectedattention layers for efficient adaptation in multi-task learning.13





Sun, M., Liu, Z., Bair, A., and Kolter, J. Z. (2023a). A simple andeffective pruning approach for large language models. arXivpreprint arXiv:2306.11695. 11





Sun, X., Choi, J., Chen, C.-Y., Wang, N., Venkataramani, S.,Srinivasan, V. V., Cui, X., Zhang, W., and Gopalakrishnan, K.(2019a). Hybrid 8-bit floating point (hfp8) training and infer-ence for deep neural networks. Advances in neural informationprocessing systems, 32. 24





Sun, Y., Dong, L., Patra, B., Ma, S., Huang, S., Benhaim, A.,Chaudhary, V., Song, X., and Wei, F. (2022). A length-extrapolatable transformer. 28





Sun, Z., Li, Z., Wang, H., He, D., Lin, Z., and Deng, Z. (2019b).Fast structured decoding for sequence models. Advances inNeural Information Processing Systems, 32. 19





Sun, Z., Suresh, A. T., Ro, J. H., Beirami, A., Jain, H., and Yu, F.(2023b). Spectr: Fast speculative decoding via optimal trans-port. arXiv preprint arXiv:2310.15141. 17





Sundar Pichai, D. H. (2024). Our next-generation model: Gemini1.5. 9





Tambe, T., Yang, E.-Y., Wan, Z., Deng, Y., Reddi, V. J., Rush, A.,Brooks, D., and Wei, G.-Y. (2020). Algorithm-hardware co-design of adaptive floating-point encodings for resilient deeplearning inference. In 2020 57th ACM/IEEE Design AutomationConference (DAC), pages 1–6. IEEE. 25





Team, G., Anil, R., Borgeaud, S., Wu, Y., Alayrac, J.-B., Yu, J.,Soricut, R., Schalkwyk, J., Dai, A. M., Hauth, A., et al. (2023).Gemini: a family of highly capable multimodal models. arXivpreprint arXiv:2312.11805. 26





Tesla (2023). Tesla dojo technology: A guide to tesla’s con-figurable floating point formats & arithmetic. https://cdn.motor1.com/pdf-files/535242876-tesla-dojo-technology.pdf/. 24





Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Roziere, B., Goyal, N., Hambro, E., Azhar, F.,`et al. (2023a). Llama: Open and efficient foundation languagemodels. arXiv preprint arXiv:2302.13971. 1, 6





Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A.,Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S.,et al. (2023b). Llama 2: Open foundation and fine-tuned chatmodels. arXiv preprint arXiv:2307.09288. 1, 16





Vaidya, N., Oh, F., and Comly, N. (2023). Optimizing inferenceon large language models with nvidia tensorrt-llm, now publiclyavailable. 20





Valipour, M., Rezagholizadeh, M., Kobyzev, I., and Ghodsi, A.(2022). Dylora: Parameter efficient tuning of pre-trained mod-els using dynamic search-free low-rank adaptation. arXivpreprint arXiv:2210.07558. 10





Wang, P., Wang, Z., Li, Z., Gao, Y., Yin, B., and Ren, X.(2023). Scott: Self-consistent chain-of-thought distillation.arXiv preprint arXiv:2305.01879. 12





Wang, W., Chen, W., Luo, Y., Long, Y., Lin, Z., Zhang, L., Lin, B.,Cai, D., and He, X. (2024a). Model compression and efficientinference for large language models: A survey. arXiv preprintarXiv:2402.09748. 3





Wang, W., Dong, L., Cheng, H., Liu, X., Yan, X., Gao, J., andWei, F. (2024b). Augmenting language models with long-termmemory. Advances in Neural Information Processing Systems,36. 27





Wang, Y., Tian, F., He, D., Qin, T., Zhai, C., and Liu, T.-Y. (2019).Non-autoregressive machine translation with auxiliary regular-ization. In Proceedings of the AAAI conference on artificialintelligence, volume 33, pages 5377–5384. 19





Wei, B., Wang, M., Zhou, H., Lin, J., Xie, J., and Sun, X. (2019).Imitation learning for non-autoregressive neural machine trans-lation. arXiv preprint arXiv:1906.02041. 19





Wei, H., Kong, L., Chen, J., Zhao, L., Ge, Z., Yu, E., Sun, J., Han,C., and Zhang, X. (2024). Small language model meets withreinforced vision vocabulary. arXiv preprint arXiv:2401.12503.26





Wu, M., Waheed, A., Zhang, C., Abdul-Mageed, M., and Aji, A. F.(2023a). Lamini-lm: A diverse herd of distilled models fromlarge-scale instructions. arXiv preprint arXiv:2304.14402. 12





Wu, Q., Lan, Z., Qian, K., Gu, J., Geramifard, A., and Yu, Z.(2020). Memformer: A memory-augmented transformer forsequence modeling. arXiv preprint arXiv:2010.06891. 27





Wu, S., Chen, H., Quan, X., Wang, Q., and Wang, R. (2023b).Ad-kd: Attribution-driven knowledge distillation for languagemodel compression. arXiv preprint arXiv:2305.10010. 12





Wu, X., Xia, H., Youn, S., Zheng, Z., Chen, S., Bakhtiari, A.,Wyatt, M., He, Y., Ruwase, O., Song, L., et al. (2023c). Ze-roquant $( 4 + 2 )$ : Redefining llms quantization with a new fp6-centric strategy for diverse generative tasks. arXiv preprintarXiv:2312.08583. 24





Wu, X., Yao, Z., and He, Y. (2023d). Zeroquant-fp: A leap forwardin llms post-training w4a8 quantization using floating-point for-mats. arXiv preprint arXiv:2307.09782. 9, 24





Wu, Y., Rabe, M. N., Hutchins, D., and Szegedy, C. (2022). Mem-orizing transformers. arXiv preprint arXiv:2203.08913. 27





Xia, H., Zheng, Z., Li, Y., Zhuang, D., Zhou, Z., Qiu, X., Li, Y.,Lin, W., and Song, S. L. (2023a). Flash-llm: Enabling cost-effective and highly-efficient large generative model inferencewith unstructured sparsity. arXiv preprint arXiv:2309.10285.11





Xia, H., Zheng, Z., Wu, X., Chen, S., Yao, Z., Youn, S., Bakhtiari,A., Wyatt, M., Zhuang, D., Zhou, Z., et al. (2024). Fp6-llm:Efficiently serving large language models through fp6-centricalgorithm-system co-design. arXiv preprint arXiv:2401.14112.24





Xia, M., Gao, T., Zeng, Z., and Chen, D. (2023b). Sheared llama:Accelerating language model pre-training via structured prun-ing. arXiv preprint arXiv:2310.06694. 11





Xiao, G., Lin, J., Seznec, M., Wu, H., Demouth, J., and Han,S. (2023a). Smoothquant: Accurate and efficient post-trainingquantization for large language models. In International Con-ference on Machine Learning, pages 38087–38099. PMLR. 9





Xiao, G., Tian, Y., Chen, B., Han, S., and Lewis, M. (2023b).Efficient streaming language models with attention sinks. arXivpreprint arXiv:2309.17453. 27





Xiao, Y., Wu, L., Guo, J., Li, J., Zhang, M., Qin, T., and Liu, T.-y.(2023c). A survey on non-autoregressive generation for neuralmachine translation and beyond. IEEE Transactions on PatternAnalysis and Machine Intelligence. 19





Xin, J., Tang, R., Lee, J., Yu, Y., and Lin, J. (2020). Deebert:Dynamic early exiting for accelerating bert inference. 13





Xiong, W., Liu, J., Molybog, I., Zhang, H., Bhargava, P., Hou, R.,Martin, L., Rungta, R., Sankararaman, K. A., Oguz, B., Khabsa,M., Fang, H., Mehdad, Y., Narang, S., Malik, K., Fan, A., Bhos-ale, S., Edunov, S., Lewis, M., Wang, S., and Ma, H. (2023).Effective long-context scaling of foundation models. 28





Xu, D., Yin, W., Jin, X., Zhang, Y., Wei, S., Xu, M., and Liu, X.(2023a). Llmcad: Fast and scalable on-device large languagemodel inference. arXiv preprint arXiv:2309.04255. 17





Xu, M., Xu, Y. L., and Mandic, D. P. (2023b). Tensorgpt: Efficientcompression of the embedding layer in llms based on the tensor-train decomposition. arXiv preprint arXiv:2307.00526. 13





Xu, X., Li, M., Tao, C., Shen, T., Cheng, R., Li, J., Xu, C., Tao,D., and Zhou, T. (2024). A survey on knowledge distillation oflarge language models. 11





Xu, Y., Xie, L., Gu, X., Chen, X., Chang, H., Zhang, H., Chen, Z.,Zhang, X., and Tian, Q. (2023c). Qa-lora: Quantization-awarelow-rank adaptation of large language models. arXiv preprintarXiv:2309.14717. 10





Yang, N., Ge, T., Wang, L., Jiao, B., Jiang, D., Yang, L., Ma-jumder, R., and Wei, F. (2023). Inference with reference:Lossless acceleration of large language models. arXiv preprintarXiv:2304.04487. 18





Yao, Z., Yazdani Aminabadi, R., Zhang, M., Wu, X., Li, C., andHe, Y. (2022). Zeroquant: Efficient and affordable post-trainingquantization for large-scale transformers. Advances in NeuralInformation Processing Systems, 35:27168–27183. 8





Yi, R., Guo, L., Wei, S., Zhou, A., Wang, S., and Xu, M. (2023).Edgemoe: Fast on-device inference of moe-based large lan-guage models. arXiv preprint arXiv:2308.14352. 16





Yin, L., Wu, Y., Zhang, Z., Hsieh, C.-Y., Wang, Y., Jia, Y., Pech-enizkiy, M., Liang, Y., Wang, Z., and Liu, S. (2023a). Outlierweighed layerwise sparsity (owl): A missing secret sauce forpruning llms to high sparsity. arXiv preprint arXiv:2310.05175.11





Yin, S., Fu, C., Zhao, S., Li, K., Sun, X., Xu, T., and Chen, E.(2023b). A survey on multimodal large language models. arXivpreprint arXiv:2306.13549. 26





Yu, G.-I., Jeong, J. S., Kim, G.-W., Kim, S., and Chun, B.-G.(2022). Orca: A distributed serving system for {Transformer-Based} generative models. In 16th USENIX Symposium on Op-erating Systems Design and Implementation (OSDI 22), pages521–538. 21





Yuan, Z., Li, Z., and Sun, L. (2023a). Tinygpt-v: Efficientmultimodal large language model via small backbones. arXivpreprint arXiv:2312.16862. 26





Yuan, Z., Liu, J., Wu, J., Yang, D., Wu, Q., Sun, G., Liu, W.,Wang, X., and Wu, B. (2023b). Benchmarking the reliabilityof post-training quantization: a particular focus on worst-caseperformance. arXiv preprint arXiv:2303.13003. 26





Yuan, Z., Niu, L., Liu, J., Liu, W., Wang, X., Shang, Y., Sun, G.,Wu, Q., Wu, J., and Wu, B. (2023c). Rptq: Reorder-based post-training quantization for large language models. arXiv preprintarXiv:2304.01089. 9, 25





Yuan, Z., Shang, Y., Song, Y., Wu, Q., Yan, Y., and Sun, G.(2023d). Asvd: Activation-aware singular value decomposi-tion for compressing large language models. arXiv preprintarXiv:2312.05821. 12





Yue, Y., Yuan, Z., Duanmu, H., Zhou, S., Wu, J., and Nie,L. (2024). Wkvquant: Quantizing weight and key/valuecache for large language models gains more. arXiv preprintarXiv:2402.12065. 10





Yuksel, S. E., Wilson, J. N., and Gader, P. D. (2012). Twenty yearsof mixture of experts. IEEE Transactions on Neural Networksand Learning Systems, 23(8):1177–1193. 15, 26





Zadeh, A. H. et al. (2020). Gobo: Quantizing attention-based nlpmodels for low latency and energy efficient inference. In MI-CRO, pages 811–824. IEEE. 24





Zeng, S., Liu, J., Dai, G., Yang, X., Fu, T., Wang, H., Ma, W.,Sun, H., Li, S., Huang, Z., et al. (2024). Flightllm: Efficientlarge language model inference with a complete mapping flowon fpga. arXiv preprint arXiv:2401.03868. 25





Zeng, Z., Hong, Y., Dai, H., Zhuang, H., and Chen, C. (2023).Consistentee: A consistent and hardness-guided early exitingmethod for accelerating language models inference. 14





Zhang, D., Yu, Y., Li, C., Dong, J., Su, D., Chu, C., and Yu, D.(2024a). Mm-llms: Recent advances in multimodal large lan-guage models. arXiv preprint arXiv:2401.13601. 26





Zhang, J., Wang, J., Li, H., Shou, L., Chen, K., Chen, G., andMehrotra, S. (2023). Draft & verify: Lossless large lan-guage model acceleration via self-speculative decoding. arXivpreprint arXiv:2309.08168. 18





Zhang, S., Roller, S., Goyal, N., Artetxe, M., Chen, M., Chen, S.,Dewan, C., Diab, M., Li, X., Lin, X. V., et al. (2022). Opt:Open pre-trained transformer language models. arXiv preprintarXiv:2205.01068. 1





Zhang, Z., Sheng, Y., Zhou, T., Chen, T., Zheng, L., Cai, R., Song,Z., Tian, Y., Re, C., Barrett, C., et al. (2024b). H2o: Heavy- ´hitter oracle for efficient generative inference of large languagemodels. Advances in Neural Information Processing Systems,36. 27





Zhao, P., Zhang, H., Yu, Q., Wang, Z., Geng, Y., Fu, F., Yang,L., Zhang, W., and Cui, B. (2024). Retrieval-augmented gen-eration for ai-generated content: A survey. arXiv preprintarXiv:2402.19473. 27





Zhao, W. X., Zhou, K., Li, J., Tang, T., Wang, X., Hou, Y., Min,Y., Zhang, B., Zhang, J., Dong, Z., et al. (2023). A survey oflarge language models. arXiv preprint arXiv:2303.18223. 1, 3





Zhong, W., Guo, L., Gao, Q., and Wang, Y. (2023). Memory-bank: Enhancing large language models with long-term mem-ory. arXiv preprint arXiv:2305.10250. 27





Zhou, B., Hu, Y., Weng, X., Jia, J., Luo, J., Liu, X., Wu, J., andHuang, L. (2024). Tinyllava: A framework of small-scale largemultimodal models. arXiv preprint arXiv:2402.14289. 26





Zhou, J., Wu, J., Gao, Y., Ding, Y., Tao, C., Li, B., Tu, F., Cheng,K.-T., So, H. K.-H., and Wong, N. (2023a). Dybit: Dynamicbit-precision numbers for efficient quantized neural network in-ference. arXiv preprint arXiv:2302.12510. 24





Zhou, W., Jiang, Y. E., Cui, P., Wang, T., Xiao, Z., Hou, Y.,Cotterell, R., and Sachan, M. (2023b). Recurrentgpt: In-teractive generation of (arbitrarily) long text. arXiv preprintarXiv:2305.13304. 27





Zhou, W., Xu, C., Ge, T., McAuley, J., Xu, K., and Wei, F. (2020).Bert loses patience: Fast and robust inference with early exit.13





Zhou, Y., Lyu, K., Rawat, A. S., Menon, A. K., Rostamizadeh,A., Kumar, S., Kagy, J.-F., and Agarwal, R. (2023c). Distill-spec: Improving speculative decoding via knowledge distilla-tion. arXiv preprint arXiv:2310.08461. 18





Zhou, Z., Li, C., Yang, F., and Suny, G. (2023d). Dimm-link:Enabling efficient inter-dimm communication for near-memoryprocessing. In 2023 IEEE International Symposium on High-Performance Computer Architecture (HPCA), pages 302–316.IEEE. 23





Zhu, W. (2021). LeeBERT: Learned early exit for BERT withcross-level optimization. In Zong, C., Xia, F., Li, W., and Nav-igli, R., editors, Proceedings of the 59th Annual Meeting ofthe Association for Computational Linguistics and the 11th In-ternational Joint Conference on Natural Language Processing(Volume 1: Long Papers), pages 2968–2980, Online. Associa-tion for Computational Linguistics. 13





Zhu, X., Li, J., Liu, Y., Ma, C., and Wang, W. (2023). A survey onmodel compression for large language models. arXiv preprintarXiv:2308.07633. 2





Zhu, Y., Zhu, M., Liu, N., Ou, Z., Mou, X., and Tang, J. (2024).Llava-phi: Efficient multi-modal assistant with small languagemodel. arXiv preprint arXiv:2401.02330. 26





Zoph, B., Bello, I., Kumar, S., Du, N., Huang, Y., Dean, J.,Shazeer, N., and Fedus, W. (2022). St-moe: Designing sta-ble and transferable sparse expert models. arXiv preprintarXiv:2202.08906. 16

