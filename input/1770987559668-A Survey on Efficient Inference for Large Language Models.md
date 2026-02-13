# A Survey on Efficient Inference for LargeLanguage Models

Zixuan Zhou*, Xuefei Ning*, Ke Hong*, Tianyu Fu, Jiaming Xu, Shiyao Li,Yuming Lou, Luning Wang, Zhihang Yuan, Xiuhong Li, Shengen Yan, Guohao Dai,Xiao-Ping Zhang Fellow, IEEE, Huazhong Yang Fellow, IEEE, Yuhan Dong, Yu Wang Fellow, IEEE

Abstract—Large Language Models (LLMs) have attracted extensiveattention due to their remarkable performance across various tasks.However, the substantial computational and memory requirements ofLLM inference pose challenges for deployment in resource-constrainedscenarios. Efforts within the field have been directed towards developingtechniques aimed at enhancing the efficiency of LLM inference. Thispaper presents a comprehensive survey of the existing literature onefficient LLM inference. We start by analyzing the primary causes ofthe inefficient LLM inference, i.e., the large model size, the quadratic-complexity attention operation, and the auto-regressive decoding ap-proach. Then, we introduce a comprehensive taxonomy that organizesthe current literature into data-level, model-level, and system-level op-timization. Moreover, the paper includes comparative experiments onrepresentative methods within critical sub-fields to provide quantitativeinsights. Last but not least, we provide some knowledge summary anddiscuss future research directions.

# 1 INTRODUCTION

Large Language Models (LLMs) have garnered substantialattention from both academia and industry in recent years.The field of LLMs has experienced notable growth and sig-nificant achievements. Numerous open-source LLMs haveemerged, including the GPT-series (GPT-1 [1], GPT-2 [2],and GPT-3 [3]), OPT [4], LLaMA-series (LLaMA [5], LLaMA2 [5], Baichuan 2 [6], Vicuna [7], LongChat [8]), BLOOM [9],FALCON [10], GLM [11], and Mistral [12], which are usedfor both academic research and commercial purposes. Thesuccess of LLMs stems from their robust capability in han-dling diverse tasks such as neural language understanding

Z. Zhou, K. Hong, T. Fu, S. Li, L. Wang are with Infinigence-AI and theDepartment of Electronic Engineering, Tsinghua University, China.E-mail: zhouzx21@mails.tsinghua.edu.cn (Z. Zhou)

X. Ning, Y. Lou, H. Yang, Y. Wang are with the Department of ElectronicEngineering, Tsinghua University, China.E-mail: foxdoraame@gmail.com (X. Ning), yu-wang@tsinghua.edu.cn (Y.Wang)

J. Xu, G. Dai are with Infinigence-AI and the Department of ElectronicEngineering, Shanghai Jiaotong University, China.E-mail: daiguohao@sjtu.edu.cn (G. Dai)

X.-P. Zhang, Y. Dong are with Tsinghua Shenzhen International GraduateSchool.E-mail: xpzhang@ieee.org (X.-P. Zhang), dongyuhan@sz.tsinghua.edu.cn(Y. Dong)

• Z. Yuan, S. Yan are with Infinigence-AI.

• X. Li is with Peking University.

• Corresponding authors: Yu Wang, Xuefei Ning, Guohao Dai.

• *Equal contribution.

(NLU), neural language generation (NLG), reasoning [13],[14], and code generation [15], consequently enabling im-pactful applications like ChatGPT, Copilot, and Bing. Thereis a growing belief [16] that the rise and achievements ofLLMs signify a significant stride towards Artificial GeneralIntelligence (AGI) for humanity.

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-11/924f71b2-d0c7-4bf4-8165-7633ffb6c93d/99e583b88b38493c876ac2cf8263ca188362dbf6a4f5e0aa2c6c57759c29df34.jpg)



Fig. 1. The challenges of LLM deployment.


However, the deployment of LLMs is not always goingsmoothly. As shown in Fig. 1, LLMs typically demandhigher computational cost, memory access cost and memoryusage in their inference process (we will analyse the rootcauses in the Sec. 2.3), which deteriorates the efficiencyindicators (e.g., latency, throughput, power consumptionand storage) in the resource-constrained scenarios. Thisposes challenges for the application of LLMs in both edgeand cloud scenarios. For example, the immense storage re-quirements render the deployment of a 70-billion-parametermodel impractical on personal laptops for tasks such asdevelopment assistance. Additionally, the low throughputwould result in significant costs if LLMs are used for everysearch engine request, leading to a considerable reductionin the profits of the search engine.

Fortunately, a substantial array of techniques has beenproposed to enable efficient inference for LLMs. To gaina comprehensive understanding of existing studies andinspire further research, this survey employs a hierarchicalclassification and systematic summarization of the currentlandscape of efficient LLM inference. Specifically, we cat-egorize relevant studies into three levels: data-level opti-mization, model-level optimization, and system-level op-timization (refer to Sec. 3 for elaboration). Moreover, weconduct experimental analyses on representative methods

within critical sub-fields to consolidate knowledge, offerpractical recommendations, and provide guidance for futureresearch endeavors.


TABLE 1Comparison of existing surveys.


<table><tr><td rowspan="2">Survey</td><td colspan="3">Optimization Levels</td><td rowspan="2">Experimental Analysis</td></tr><tr><td>Data-level</td><td>Model-level</td><td>System-level</td></tr><tr><td>[17], [18], [19]</td><td></td><td>✓</td><td></td><td></td></tr><tr><td>[20]</td><td></td><td>✓</td><td></td><td>✓</td></tr><tr><td>[21]</td><td>✓</td><td>✓</td><td></td><td></td></tr><tr><td>[22]</td><td></td><td>✓</td><td>✓</td><td></td></tr><tr><td>[23], [24]</td><td>✓</td><td>✓</td><td>✓</td><td></td></tr><tr><td>Ours</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td></tr></table>

Currently, several surveys [17], [18], [19], [20], [21], [22],[23] have been conducted in the field of efficient LLMs.These surveys primarily focus on different aspects of LLMefficiency but offer opportunities for further improvement.Zhu et al. [17], Park et al. [18], Wang et al. [19] and Tanget al. [20] concentrate on model compression techniqueswithin model-level optimization. Ding et al. [21] centeron efficiency research considering both data and modelarchitecture perspectives. Miao et al. [22] approach efficientLLM inference from a machine learning system (MLSys) re-search perspective. In contrast, our survey provides a morecomprehensive research scope, addressing optimization atthree levels: data-level, model-level, and system-level, withthe inclusion of recent advancements. While Wan et al. [23]and Xu et al. [24] also deliver comprehensive review ofefficient LLM research, our work extends by incorporatingcomparative experiments and offering practical insights andrecommendations based on experimental analyses in sev-eral critical sub-fields like model quantization and servingsystems. A comparison of these surveys is summarized inTable 1.

The remainder of this survey is organized as follows:Sec. 2 introduces the basic concept and knowledge aboutLLMs and presents a detailed analysis of the efficiencybottlenecks during the inference process of LLMs. Sec. 3demonstrates our taxonomy. Sec. 4 to Sec. 6 respectivelypresent and discuss studies on efficiency optimization atthree distinct levels. Sec. 7 offers broader discussions forseveral key application scenarios. Sec. 8 concludes the keycontributions provided by this survey.

# 2 PRELIMINARIES

# 2.1 Transformer-Style LLMs

Language modeling, as the fundamental function of lan-guage models (LMs), involves modeling the likelihood ofthe word sequence and predicting the distribution of subse-quent words. Over recent years, researchers have discoveredthat scaling up language models not only enhances theirlanguage modeling ability but also engenders emergentcapabilities for tackling more intricate tasks beyond conven-tional NLP tasks [25]. These scaled-up language models arereferred to as large language models (LLMs).

The mainstream LLMs are designed based on the Trans-former architecture [26]. Specifically, a typical Transformer

architecture is composed of several stacked Transformerblocks. Typically, a Transformer block consists of a Multi-Head Self-Attention (MHSA) block, a Feed Forward Net-work (FFN), and a LayerNorm (LN) operation. For eachblock, it receives the output features of the previous oneas the input, and passes the features through each sub-module to obtain the output. Specially, before the first block,a tokenizer is used to convert the original input sentenceinto a sequence of tokens, and a following embedding layerserves to convert the tokens into the input features. Then,the additional position embeddings are added into the inputfeatures to encode the sequential order of each input token.

The core concept of the Transformer architecture is theself-attention mechanism, which is adopted in the MHSAblock. Specifically, denoted the input features as $\boldsymbol { X } \ =$$[ x _ { 1 } , x _ { 2 } , . . . , x _ { n } ] ,$ the MHSA block applies linear projection tothem and obtains a set of queries Q, keys K and values V asEq. 1:

$$
Q _ {i} = X W ^ {Q _ {i}}, K _ {i} = X W ^ {K _ {i}}, V _ {i} = X W ^ {V _ {i}}, \tag {1}
$$

where $W ^ { Q _ { i } }$ , $W ^ { K _ { i } }$ and $W ^ { V _ { i } }$ are the projection matricescorresponding to the $i$ -th attention head. Then the self-attention operation is applied to each tuple of $( Q _ { i } , K _ { i } , V _ { i } )$and get the feature of the $i$ -th attention head $Z _ { i }$ as Eq. 2:

$$
Z _ {i} = \operatorname {A t t e n t i o n} \left(Q _ {i}, K _ {i}, V _ {i}\right) = \operatorname {S o f t m a x} \left(\frac {Q _ {i} K _ {i} ^ {T}}{\sqrt {d _ {k}}}\right) V _ {i}, \tag {2}
$$

where $d _ { k }$ is the dimension of the queries (keys). Note thatthe self-attention operation contains the matrix multipli-cation operation, its computation complexity is quadraticin the input length. Finally, the MHSA block concatenatesthe features of all the attention heads and applies a linearprojection to them to form its output $Z$ as Eq. 3:

$$
Z = \operatorname {C o n c a t} \left(Z _ {1}, Z _ {2}, \dots , Z _ {h}\right) W ^ {O}, \tag {3}
$$

where $W _ { O }$ is the projection matrix. As can be seen, theself-attention mechanism allows the model to identify theimportance of different input parts regardless of the dis-tance, and thus can capture the long-range dependenciesand complex relationships in the input sentence.

Another important module in the Transformer block isthe FFN. Typically, FFN is placed after the MHSA blockand consists of two linear transformation layers with a non-linear activation function. It receives the output features $X$from the MHSA block and processes them as Eq 4:

$$
\operatorname {F F N} (X) = W _ {2} \sigma \left(W _ {1} X\right), \tag {4}
$$

where $W _ { 1 }$ and $W _ { 2 }$ denote the weight matrices of the twolinear layers, and $\sigma ( \cdot )$ denotes the activation function.

# 2.2 Inference Process of LLMs

The most popular LLMs, i.e., decoder-only LLMs, oftenadopt the auto-regressive method to generate the outputsentence. Specifically, the auto-regressive method generatesthe tokens one by one. In each generation step, the LLMtakes as input the whole token sequences, including the in-put tokens and previously generated tokens, and generatesthe next token. With the increase in sequence length, thetime cost of the generation process grows rapidly. To ad-dress this challenge, a crucial technique, namely key-value

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-11/924f71b2-d0c7-4bf4-8165-7633ffb6c93d/6fa14fb087255b16aa55e35830834e0b11a968f8a67dd53b8a9ccde4d65432aa.jpg)



Fig. 2. Demonstration of the prefilling stage (a) and decoding stage (b).


(KV) cache, has been introduced to expedite the generationprocess. The KV cache technique, as its name suggests,involves storing and reusing previous key (K) and value (V)pairs within the Multi-Head Self-Attention (MHSA) block.This technique has been widely adopted in LLM inferenceengines and systems due to its substantial optimizationof generation latency. Based on the above methods andtechniques, the inference process of LLMs can be dividedinto two stages:

• Prefilling Stage: The LLM calculates and stores the KVcache of the initial input tokens, and generates the firstoutput token, as shown in Fig. 2(a).

• Decoding Stage: The LLM generates the output tokensone by one with the KV cache, and then updates it withthe key (K) and value (V) pairs of the newly generatedtoken, as shown in Fig. 2(b).

As shown in Fig. 3, we illustrate some critical efficiencyindicators. As for the latency, we denote first token latencyas the latency to generate the first output token in theprefilling stage, while we denote per-output token latencyas the average latency to generate one output token inthe decoding stage. Besides, we use generation latencyto denote the latency to generate the whole output tokensequences. As for the memory, we use model size to denotethe memory to store the model weights, and use KV cachesize to denote the memory to store the KV cache. Addition-ally, peak memory denotes the maximum memory usageduring the generation process, which is approximately equalto the memory sum of model weights and KV cache. Apartfrom the latency and memory, throughput is also a widely-used indicator in the LLM serving system. We use tokenthroughput to denote the number of generated tokens persecond, and use request throughput to denote the numberof completed requests per second.

# 2.3 Efficiency Analysis

Deploying LLMs on resource-constrained scenarios whilepreserving their powerful capabilities poses a significantchallenge for both practitioners and researchers. For in-stance, let’s consider to deploy a LLaMA-2-70B model,which contains 70 billion parameters. Storing its weightsin FP16 format necessitates 140 GB of VRAM, requiringat least 6 RTX 3090Ti GPUs (each with 24 GB VRAM)

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-11/924f71b2-d0c7-4bf4-8165-7633ffb6c93d/2f2e431274c6111fcf93cf0af518247a5f0a7545b066aacc8141708d87347563.jpg)



Fig. 3. Illustration of the memory variation through time (latency) duringone generation process. Note that we ignore the activation size in thisfigure for a simplification.


or 2 NVIDIA A100 GPUs (each with 80 GB VRAM) forinference. As for latency, generating one token on 2 NVIDIAA100 GPUs requires approximately 100 milliseconds. Con-sequently, generating a sequence with hundreds of tokensrequires more than 10 seconds. In addition to storage andlatency, the efficiency indicators, such as throughput, energyand power consumption, also need to be considered. Duringthe LLM inference process, three important factors wouldlargely affect these indicators, i.e., the computational cost,the memory access cost and the memory usage. Yuan etal. [27] provide a more systematic analysis to demonstratehow these factors affect the inference inefficiency with aroofline model. In the following, we further analyze threeroot causes of inefficiency in the LLM inference process,focusing on the above three key factors:

• Model Size: Mainstream LLMs typically incorporatebillions or even trillions of parameters. For instance,the LLaMA-70B model comprises 70 billion parame-ters, while the GPT-3 model scales up to 175 billionparameters. This considerable model size contributessignificantly to the elevated computational cost, mem-ory access cost, and memory usage during the LLMinference process.

• Attention Operation: As illustrated in Sec. 2.1 andSec. 2.2, in the prefilling stage, the self-attention oper-ation exhibits quadratic computational complexity inthe input length. Consequently, as the input lengthincreases, the computational cost, memory access cost,and memory usage of the attention operation escalaterapidly.

• Decoding Approach: The auto-regressive decoding ap-proach generates the tokens one by one. In each decod-ing step, all the model weights are loaded from the off-chip HBM to the GPU chip, leading to a large memoryaccess cost. In addition, the size of KV cache increaseswith the growth in the input length, potentially leadingto fragmented memory and irregular memory accesspatterns.

# 3 TAXONOMY

In the aforementioned discussion, we identify key factors(i.e., computational cost, memory access cost and mem-ory usage) that significantly impact the efficiency during

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-11/924f71b2-d0c7-4bf4-8165-7633ffb6c93d/ffeafc3bc9b684001da60295164c68c2fd24a3f9bd86146dc8b4f6bda9e7471b.jpg)



Fig. 4. Taxonomy of efficient inference methods for Large Language Models.


the LLM inference process, and further analyze three rootcauses (i.e., model size, attention operation and decodingapproach). Many efforts have been made to optimize theinference efficiency from different perspectives. By carefullyreviewing and summarizing these studies, we classify theminto three levels, i.e., data-level optimization, model-leveloptimization and system-level optimization (as shown inFig. 4):

• Data-level Optimization refers to improving the ef-ficiency via optimizing the input prompts (i.e., inputcompression) or better organizing the output content(i.e., output organization). This line of optimization typ-ically does not change the original model, thus is freeof costly model training cost (note that a small amount

of training for auxiliary models might be required, butthis cost can be ignored compared with the training costfor original LLMs).

• Model-level Optimization refers to designing an ef-ficient model structure (i.e., efficient structure design)or compressing the pre-trained models (i.e., modelcompression) in the inference process to improve itsefficiency. This line of optimization (1) often requirescostly pre-training or a smaller amount of fine-tuningcost to retain or recover the model ability, and (2) istypically lossy in the model performance.

• System-level Optimization refers to optimizing theinference engine or the serving system. This line of opti-mization (1) does not involve the costly model training,

and (2) is typically lossless in model performance1. Inaddition, we provide a brief introduction for hardwareaccelerator design in Sec. 6.3.

# 4 DATA-LEVEL OPTIMIZATION

In the data level, prior studies can be divided into twocategories, i.e., input compression and output organization.Input compression techniques directly shorten the model in-put to reduce the inference cost. While output organizationtechniques enable batch (parallel) inference via organizingthe structure of output content, which can improve thehardware utilization and reduce the generation latency.

# 4.1 Input Compression

In the practical application of LLMs, prompts are crucial.Numerous studies suggest new ways to design promptseffectively and show in practice that well-designed promptscan unleash the capabilities of LLMs. For instance, In-Context Learning (ICL) [45] suggests to include multiplerelevant examples within the prompt. This approach en-courages LLMs to learn through analogy. Chain-of-Thought(CoT) [14] proposes to incorporate a sequence of intermedi-ate reasoning steps within the in-context examples, whichhelp LLMs to conduct complex reasoning. However, theseprompting techniques inevitably lead to longer prompts,which poses a challenge because the computational cost andmemory usage increase quadratically during the prefillingstage (as illustrated in Sec. 2.3).

To address this challenge, input prompt compres-sion [33] has been proposed to shorten prompts withoutsignificantly impacting the quality of answers from LLMs.Within this field, relevant studies are categorized into fourgroups, as depicted in Figure 5: prompt pruning, promptsummary, soft prompt-based compression, and retrieval-augmented generation.

# 4.1.1 Prompt Pruning

The core idea behind the prompt pruning is to removeunimportant tokens, sentences, or documents online fromeach input prompt based on predefined or learnable impor-tance indicators. DYNAICL [38] proposes to dynamicallydecide the optimal number of in-context examples for agiven input based on the computational budget via a well-trained LLM-based meta controller. Selective Context [39]proposes to merge tokens into units, and then applies aunit-level prompt pruning based on the self-informationindicator (i.e., negative log likelihood). STDC [40] prunesthe prompts based on the parse tree, which iterativelyremoves phrase nodes that cause the smallest performancedrop after pruning it. PCRL [41] introduces a token-levelpruning scheme based on reinforcement learning. The mainidea behind PCRL is to train a policy LLM by combiningfaithfulness and compression ratio into the reward func-tion. Faithfulness is measured as the output similarity be-tween the compressed prompt and the original prompt.RECOMP [36] implements a sentence-level pruning strategy

to compress prompts for Retrieval-Augmented LanguageModels (RALMs). The approach involves encoding the in-put question and documents into latent embeddings usinga pre-trained encoder. Then, it decides which documentsto remove based on the similarity of their embeddingswith the question’s embedding. LLMLingua [42] introducesa coarse-to-fine pruning scheme for prompt compression.Initially, it performs a demonstration-level pruning followedby token-level pruning based on perplexity. To enhanceperformance, LLMLingua proposes a budget controller thatdynamically allocates the pruning budget across differentparts of prompts. Additionally, it utilizes an iterative token-level compression algorithm to address inaccuracies in-troduced by conditional independence assumptions. Fur-thermore, LLMLingua incorporates a distribution align-ment strategy to align the output distribution of the targetLLM with a smaller LLM used for perplexity calculation.LongLLMLingua [43] builds upon LLMLingua with severalenhancements: (1) It utilizes perplexity conditioned on theinput question as the indicator for prompt pruning. (2) Itallocates varying pruning ratios to different demonstrationsand reorders the demonstrations within the final promptbased on their indicator values. (3) It restores the originalcontent based on the response. CoT-Influx [44] introducesa coarse-to-grained pruning method for Chain-of-Thought(CoT) prompts using reinforcement learning. Specifically, itprunes unimportant examples, followed by pruning unim-portant tokens within the remaining examples.

# 4.1.2 Prompt Summary

The core idea of prompt summary is to condense theoriginal prompt into a shorter summary while preservingsimilar semantic information. These techniques also serve asonline compression methods for prompts. In contrast to theaforementioned prompt pruning techniques that preservethe unpruned tokens unchanged, this line of methods con-verts the entire prompt into its summation. RECOMP [36]introduces an Abstractive Compressor that takes an inputquestion and retrieved documents as input, and producesa concise summary. Specifically, it distills a lightweightcompressor from the extreme-scale LLMs to perform thesummary. SemanticCompression [37] proposes a semanticcompression method. It starts by breaking down the textinto sentences. Next, it groups sentences together by topicand then summarizes the sentences within each group.

# 4.1.3 Soft Prompt-based Compression

The core idea of this kind of compression techniques is todesign a soft prompt, significantly shorter than the orig-inal prompt, for use as input to LLMs. The soft promptis defined as a sequence of learnable continuous tokens.Some techniques adopt offline compression for the fixedprefix prompt (e.g., system prompt, task-specific prompt).For example, PromptCompression [33] trains a soft promptto emulate a predetermined system prompt. The approachinvolves adding several soft tokens before the input tokensand enabling these soft tokens to be adjusted during back-propagation. Following fine-tuning on the prompt dataset,the sequence of soft tokens serves as the soft prompt.Gisting [34] introduces a method to condense task-specific

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-11/924f71b2-d0c7-4bf4-8165-7633ffb6c93d/dda40ec027ee053e0cd5362ab77bf56d256082aee64d26106db85c48f2f1b451.jpg)



Fig. 5. Taxonomy of the input compression methods for Large Language Models.


prompts into a concise set of gist tokens using prefix-tuning [46]. Given that task-specific prompts differ acrosstasks, prefix-tuning is applied individually for each task.To enhance efficiency, Gisting further introduces a meta-learning approach that predicts gist tokens for new unseentasks based on the gist tokens of previous tasks.

Other techniques adopt online compression for everynew input prompts. For instance, AutoCompressors [30]train a pre-trained LM to compress the prompts into sum-mary vectors via unsupervised learning. ICAE [35] trainsan autoencoder to compress the original context into shortmemory slots. Specifically, ICAE employs a LoRA-adaptedLLM as the encoder, and uses the target LLM as the decoder.A set of memory tokens is added before the input tokensand encoded into memory slots.

# 4.1.4 Retrieval-Augmented Generation

Retrieval-Augmented Generation (RAG) [29] aims to im-prove the quality of LLMs’ responses by incorporating exter-nal knowledge sources. RAG can be also viewed as a tech-nique to improve the inference efficiency when handling alarge amount of data. Instead of merging all informationinto an excessively long prompt, RAG only adds relevantretrieved information to the original prompt, ensuring thatthe model receives necessary information while reducingprompt length significantly. FLARE [30] uses predictions ofupcoming sentences to proactively decide when and whatinformation to retrieve. REPLUG [31] treats the LLM as ablack box and augments it with a tuneable retrieval model.It prepends retrieved documents to the input for the frozenblack-box LLM, and further utilizes the LLM to supervisethe retrieval model. Self-RAG [32] enhances LLM’s qualityand factuality through retrieval and self-reflection. It intro-duces reflection tokens to make the LLM controllable duringthe inference phase.

# 4.2 Output Organization

The traditional generation process of LLMs is entirely se-quential, leading to significant time consumption. Outputorganization techniques aim to (partially) parallelize gener-ation via organizing the structure of output content.

Skeleton-of-Thought (SoT) [47] is pioneering in this di-rection. The core idea behind SoT is to leverage the emerg-ing ability of LLMs to plan the output content’s struc-ture. Specifically, SoT consists of two main phases. In thefirst phase (i.e., skeleton phase), SoT instructs the LLM to

generate a concise skeleton of the answer using a prede-fined ”skeleton prompt.” For instance, given a question like”What are the typical types of Chinese dishes?”, the outputat this stage would be a list of dishes (e.g., noodles, hotpot, rice) without elaborate descriptions. Then, in the secondphase (i.e., point-expanding phase), SoT instructs the LLMto expand each point in the skeleton simultaneously usinga ”point-expanding prompt,” and then concatenates theseexpansions to form the final answer. When applied to open-source models, point-expanding can be performed throughbatch inference, which optimizes hardware utilization andreduces overall generation latency using the same compu-tational resources. To mitigate the additional computation

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-11/924f71b2-d0c7-4bf4-8165-7633ffb6c93d/579830590e37573617cbb22309cfcf10686c64ce11c680a5b6d95a8a29df093b.jpg)



(a) The skeleton stage



(b) The point-expanding stage



Fig. 6. Demonstration of the inference process of SoT.


overhead brought by the extra prompt (i.e., skeleton promptand point-expanding prompt), SoT discusses the possibility ofsharing the KV cache of the common prompt prefix acrossmultiple points in the point expansion phase. Additionally,SoT uses a router model to decide whether applying SoT isappropriate for specific questions, aiming to limit its use tosuitable cases. As a result, SoT achieves up to a $2 . 3 9 \times$ speed-up on 12 recently released LLMs, and improves the answerquality for many questions by improving the diversity andrelevance of their answer.

SGD [48] further extends the idea of SoT by organizingsub-problem points into a Directed Acyclic Graph (DAG)and answering the logic-independent sub-problems in par-allel in one turn. Similar to SoT, SGD also leverages theemerging ability of LLMs to generate the output structureby providing manually-crafted prompts along with severalexamples. SGD relaxes the strict independence assumptionamong different points to enhance the quality of answers,especially for math and coding problems. Compared withSoT, SGD prioritizes answer quality over speed. Addition-ally, SGD introduces an adaptive model selection approach,

assigning an optimal model size to handle each sub-problembased on its estimated complexity, thus further improvingefficiency.

APAR [49] adopts a similar idea with SoT, leveragingLLMs to output special control tokens (i.e., [fork]) for auto-matically and dynamically triggering the parallel decoding.To effectively exploit the inherent parallelizable structurewithin the output content and accurately generate controltokens, APAR fine-tunes the LLMs on carefully-designeddata that formed in specific tree structure. As a result, APARachieves an average $1 . 4 { \sim } 2 . 0 \times$ speed-up on benchmarks andcases a negligible impact on the answer quality. Further-more, APAR combines their decoding approach with thespeculative decoding technique (i.e., Medusa [50]) and serv-ing system (i.e. vLLM [51]) to further improve the inferencelatency and system throughput, respectively.

SGLang [52] introduces a domain-specific language(DSL) in Python featuring primitives that flexibly facili-tate LLM programming. The core idea behind SGLang isto analyze dependencies among various generation callsautomatically, and perform batch inference and KV cachesharing based on this analysis. With this language, users canimplement various prompting strategies easily and benefitfrom the automatic efficiency optimization of SGLang (e.g.,SoT [47], ToT [53]). Furthermore, SGLang introduces andcombines several system-level compilation techniques, suchas code movement and prefetching annotations.

# 4.3 Knowledge, Suggestions and Future Direction

The growing demand for LLMs to handle longer inputs andgenerate longer outputs highlights the importance of thedata-level optimization techniques. Within these techniques,input compression methods primarily target enhancing theprefilling stage by diminishing the computational and mem-ory cost resulting from the attention operation. Additionally,for API-based LLMs, these methods can reduce the API costassociated with input tokens. In contrast, output organiza-tion methods concentrate on optimizing the decoding stageby alleviating the substantial memory access cost associatedwith auto-regressive decoding approach.

As LLMs become more and more capable, there is poten-tial to utilize them to compress the input prompts or struc-ture the output content. Recent advancements in outputorganization methods [47], [48], [49] demonstrate the effec-tiveness of leveraging LLMs to organize the output contentinto independent points or a dependency graph, facilitatingbatch inference for improving generation latency. Thesemethods capitalize on the inherent parallelizable structurewithin output content, enabling LLMs to perform paralleldecoding to enhance hardware utilization and thereby re-duce end-to-end generation latency.

Recently, diverse prompting pipelines (e.g., ToT [53],GoT [54]) and agent frameworks [55], [56], [57] are emerg-ing. While these innovations enhance LLMs’ capabilities,they also extend the length of inputs, leading to increasedcomputational cost. To address this challenge, adoptinginput compression techniques to reduce input length showspromise as a solution. Simultaneously, these pipelines andframeworks naturally introduce more parallelism into out-put structures, offering increased potential for parallel de-coding and key-value (KV) cache sharing across different

decoding threads. SGLang [52] supports flexible LLM pro-gramming and offers opportunities for front-end and back-end co-optimization, laying the groundwork for furtherextensions and improvements in this area. In summary,data-level optimization, including input compression andoutput organization techniques, would become increasinglynecessary to enhance efficiency in the foreseeable future.In addition to optimizing the efficiency of existing frame-works, certain studies focus on designing more efficientagent frameworks directly. For example, FrugalGPT [58]proposes a model cascade comprising LLMs of varyingsizes, with the inference process being halted early if themodel reaches a sufficient level of certainty regarding theanswer. This approach aims to achieve efficiency by leverag-ing a tiered model architecture and intelligent inference ter-mination based on model confidence estimation. Comparedwith model-level dynamic inference techniques (Sec. 5.2.5),FrugalGPT performs dynamic inference at the pipeline level.

# 5 MODEL-LEVEL OPTIMIZATION

The model-level optimization for LLM efficient inferencemainly concentrates on optimizing the model structure ordata representation. Model structure optimization involvesdirectly designing efficient model structure, modifying theoriginal model and adjusting the inference-time architec-ture. In terms of data representation optimization, the modelquantization technique is commonly employed.

In this section, we categorize model-level optimizationtechniques based on the additional training overhead theyrequire. The first category involves designing more efficientmodel structures (referred to as efficient structure design).Models developed using this approach typically requiretraining from scratch. The second category focuses on com-pressing pre-trained models (referred to as model compres-sion). Compressed models in this category generally requireonly minimal fine-tuning to restore their performance.

# 5.1 Efficient Structure Design

Currently, state-of-the-art LLMs commonly employ theTransformer architecture, as discussed in Section 2.1. How-ever, the key components of Transformer-based LLMs, in-cluding the Feed Forward Network (FFN) and attentionoperation, present efficiency challenges during inference.We identify the causes as follows:

• The FFN contributes a substantial portion of the modelparameters in Transformer-based LLMs, resulting insignificant memory access cost and memory usage,particularly during the decoding stage. For instance, theFFN module accounts for $6 3 . 0 1 \%$ of the parameters inthe LLaMA-7B model and $7 1 . 6 9 \%$ in the LLaMA-70Bmodel.

• The attention operation demonstrates quadratic com-plexity in the input length, leading to substantial com-putational cost and memory usage, especially whendealing with longer input contexts.

To tackle these efficiency challenges, several studies haveconcentrated on developing more efficient model structures.We categorize these studies into three groups (as depicted inFig. 7): efficient FFN design, efficient attention design, andTransformer alternates.

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-11/924f71b2-d0c7-4bf4-8165-7633ffb6c93d/986b34f794f6e4c1b562ce0f12dcdd5ddd429a0c9d28cc992a78f2ecb1144b62.jpg)



Fig. 7. Taxonomy of the efficient structure design for Large Language Models.


# 5.1.1 Efficient FFN Design

In this field, many studies concentrate on integrating theMixture-of-Experts (MoE) technique [98] into LLMs to en-hance their performance while maintaining the computa-tional cost. The core idea of MoE is to dynamically allocatevarying computational budgets to different input tokens. InMoE-based Transformers, multiple parallel Feed ForwardNetworks (FFNs), namely experts, are utilized alongside atrainable routing module. During inference, the model se-lectively activates specific experts for each token controlledby the routing module.

Some researches concentrate on the construction of FFNexpert, which mainly focus on optimizing the process ofacquiring expert weights or making these experts morelightweight for efficiency. For instance, MoEfication [89] de-vises a method to transform a non-MoE LLM into the MoEversion using its pre-trained weights. This approach elimi-nates the need for expensive pre-training of the MoE model.To accomplish this, MoEfication first divides FFN neuronsof the pre-trained LLM into multiple groups. Within eachgroup, the neurons are commonly activated simultaneouslyby the activation function. Then, it restructures each groupof neurons as an expert. Sparse Upcycling [91] introduces amethod to initialize the weights of MoE-based LLM directlyfrom a dense model’s checkpoint. In this approach, theexperts within the MoE-based LLM are exact replicas ofthe FFN from the dense model. By employing this straight-forward initialization, Sparse Upcycling can efficiently trainthe MoE model to achieve high performance. MPOE [90]proposes to reduce the parameters of MoE-based LLMsthrough Matrix Product Operators (MPO) decomposition.This method involves decomposing each weight matrix ofthe FFN into a global shared tensor containing commoninformation and a set of local auxiliary tensors that capturespecialized features.

Another line of researches focuses on improving thedesign of the routing module (or strategy) within MoEmodels. In previous MoE models, the routing module often

causes the load imbalance problem, which denotes thatsome experts are assigned a large number of tokens whilethe others handle only a few. This imbalance not onlywastes the capacities of the under-utilized experts, whichdegrades model performance, but also degrades the infer-ence efficiency. Current MoE implementations [88], [99],[100] often use batched matrix multiplication to computeall FFN experts simultaneously. This requires that the inputmatrices of each expert must have the same shape. However,since the load imbalance problem exists, input token setsfor these under-utilized experts are needed to be padded tomeet the shape constraint, resulting in a waste of compu-tation. Therefore, the major aim of routing module designis achieving better balance in token assignment for MoEexperts. Switch Transformers [88] introduces an additionalloss, namely the load balancing loss, into the final lossfunction to penalize imbalanced assignments by the routingmodule. This loss is formulated as the scaled dot-productbetween the token assignment fraction vector and a uniformdistribution vector. As a result, the loss is minimized onlywhen the token assignment is balanced across all experts.This approach encourages the routing module to distributetokens evenly among experts, promoting load balance andultimately improving model performance and efficiency.BASE [92] learns an embedding for each expert in an end-to-end manner and then assigns experts to tokens based onthe similarity of their embeddings. To ensure load balance,BASE formulates a linear assignment problem and utilizesthe auction algorithm [101] to solve this problem efficiently.Expert Choice [93] introduces a simple yet effective strategyto ensure perfect load balance within MoE-based models.Unlike previous methods that assign experts to tokens,Expert Choice allows each expert to independently selectthe top- $\mathbf { \nabla } \cdot k$ tokens based on their embedding similarities. Thisapproach ensures that each expert handles a fixed numberof tokens, even though each token might be assigned to adifferent number of experts.

In addition to the aforementioned researches focusing

on the model architecture itself, there are also studies thatconcentrate on improving the training methods for MoE-based models. SE-MoE [94] introduces a new auxiliary losscalled the router z-loss, which aims to enhance the stabilityof model training without compromising performance. SE-MoE identifies that the exponential functions introduced bysoftmax operations in the routing module can exacerbateroundoff errors, leading to training instability. To addressthis issue, the router z-loss penalizes large logits that are in-put into exponential functions, thereby minimizing roundofferrors during training. StableMoE [95] points out the routingfluctuation problem existing in the MoE-based LLMs, whichdenotes the inconsistency of the expert assignment in thetraining and inference stage. For the same input token, it isassigned to different experts along with training, but onlyactivates one expert at inference time. To address this issue,StableMoE suggests a more consistent training approach.It first learns a routing strategy and then keeps it fixedduring both the model backbone training and the inferencestage. SMoE-Dropout [96] designs a novel training methodfor MoE-based LLMs, which proposes to gradually increasethe number of activated experts during the training process.This approach enhances the scalability of MoE-based mod-els for inference and downstream fine-tuning. GLaM [97]pre-trains and releases a series of models with various pa-rameter sizes, demonstrating their comparable performanceto dense LLMs on few-shot tasks. The largest model in thisfamily has a parameter size of up to 1.2 trillion. Mixtral8x7B [12] is a remarkable recently released open-sourcemodel. During inference, it utilizes only 13 billion activeparameters and achieves superior performance comparedto the LLaMA-2-70B model across different benchmarks.Mixtral 8x7B consists of 8 Feed-Forward Network (FFN)experts in each layer, with each token assigned to twoexperts during inference.

# 5.1.2 Efficient Attention Design

The attention operation is a critical component in the Trans-former architecture. However, its quadratic complexity inrelation to input length leads to substantial computationalcost, memory access cost, and memory usage, especiallywhen dealing with long contexts. To address this issue,researchers are exploring more efficient approaches to ap-proximate the functionality of the original attention oper-ation. These studies can be broadly categorized into twomain branches: multi-query attention and low-complexityattention.

Multi-Query Attention. Multi-query attention (MQA) [77]optimizes the attention operation by sharing the key (K)and value (V) cache across different attention heads. Thisstrategy effectively reduces both memory access cost andmemory usage during inference, contributing to improvedefficiency in Transformer models. As introduced in Sec. 2.2,the Transformer-style LLMs typically adopts multi-headattention (MHA) operation. This operation requires stor-ing and retrieving K and V pairs for each attention headduring the decoding stage, leading to substantial increasesin memory access cost and memory usage. MQA tacklesthis challenge by using the same K and V pairs acrossdifferent heads while maintaining distinct query (Q) values.Through extensive testing, it has been demonstrated that

MQA significantly reduces memory requirements with onlya minimal impact on model performance, making it a cru-cial strategy for enhancing inference efficiency. The conceptof MQA is further extended by Grouped-query attention(GQA) [78], which can be seen as a blend of MHA andMQA. Specifically, GQA segments the attention heads intogroups, storing a single set of K and V values for eachgroup. This method not only sustains the benefits of MQAin reducing memory overhead but also offers an enhancedbalance between inference speed and output quality.

Low-Complexity Attention. Low-complexity attentionmethods aim to design new mechanisms that reduce thecomputational complexity of each attention head. To sim-plify the discussion, we assume that the dimensions of theQ (query), K (key), and V (value) matrices are identical,with $Q , K , V \in \mathbb { R } ^ { n \times d }$ . Since the following work does notinvolve altering the number of attention heads like MQA,our discussions focus on the attention mechanism withineach head. As introduced in Section 2.2, the computationalcomplexity of the conventional attention mechanism scalesas $\bar { \mathcal { O } } ( n ^ { 2 } )$ , exhibiting quadratic growth with respect to the in-put length $n$ . To address the inefficiency issue, kernel-basedattention and low-rank attention methods are proposed toreduce the complexity to ${ \mathcal { O } } ( n )$ .

• Kernel-based Attention. Kernel-based attention designskernel $\phi$ to approximate the non-linear softmax oper-ation of Softmax $( Q K ^ { T } )$ with a linear dot product be-tween kernel-transformed feature maps, i.e., ${ \dot { \phi } } ( Q ) \phi ( K ) ^ { T }$ .It avoids the conventional quadratic computation associ-ated with $Q K ^ { T } \in \mathbb { R } ^ { n \times n }$ by prioritizing the computationof $\phi ( K ) ^ { T } V \in \mathbb { R } ^ { d \times d } ,$ , followed by its multiplication with$\phi ( Q ) \in \mathbb { R } ^ { n \times d }$ . Specifically, the input Q and K matrices arefirst mapped into kernel space using a kernel function $\phi ,$while maintaining their original dimensions. Leveragingthe associative property of matrix multiplication allowsfor the multiplication of K and V prior to their interactionwith Q. The attention mechanism is reformulated as:

$$
\operatorname {S o f t m a x} \left(Q K ^ {T}\right) V \approx \phi (Q) \left(\phi (K) ^ {T} V\right), \tag {5}
$$

where $\phi ( Q ) , \phi ( K ) \in \mathbb { R } ^ { n \times d }$ . This strategy effectively re-duces the computational complexity to $\dot { \mathcal { O } } ( n d ^ { 2 } )$ , render-ing it linear with respect to the input length. LinearTransformer [84] is the first work to propose the kernel-based attention. It adopts $\phi ( x ) = \bar { \mathrm { e l u } ( x ) + 1 }$ as the ker-nel function, where $\mathrm { e l u } ( \cdot )$ denotes the exponential linearunit activation function. Performers [85] and RFA [86]proposes to use random feature projection to better ap-proximate the softmax function. PolySketchFormer [87]employs polynomial functions and sketching techniquesto approximate the softmax function.

• Low-Rank Attention. Low-Rank Attention technique em-ploys compression on the token dimensions (i.e., n) of theK and V matrices to a smaller, fixed length (i.e., $k$ ) beforeperforming the attention computation. The approach isbased on the insight that the $n \times n$ attention matrixoften exhibits a low-rank property, making it feasible tocompress it in the token dimension. The main focus ofthis line of researches is to design effective methods forthe compression, where $X$ can be context matrix or K andV matrices:

$$
X \in \mathbb {R} ^ {n \times d} \rightarrow X ^ {\prime} \in \mathbb {R} ^ {k \times d}. \tag {6}
$$

One line of work uses linear projection to compress thetoken dimension. It is done by multiplying K and V matri-ces with projection matrices $P _ { k } , P _ { v } \in \mathbb R ^ { \tilde { k } \times n }$ . In this way,the computational complexity of the attention operationis reduced to $O ( n k d ) .$ , which is linear to the input length.Linformer [79] first observes and analyses the low-rankproperty of the attention map, and proposes the low-rankattention framework. LRT [80] proposes to simultaneouslyapply low-rank transformation to both attention blockand FFN to further improve the computational efficiency.FLuRKA [81] combines the low-rank transformation andkernalization to the attention matrices to further improvethe efficiency. Specifically, it first reduces the token dimen-sion of K and V matrices, and then applies kernel functionto the Q and low-rank K matrices.

Aside from linear projection, other token-dimensioncompression methods are also proposed. Luna [82] andSet Transformer [83] leverage additional attention compu-tations alongside smaller queries to effectively compressthe K and V matrices. Luna [82] involves an extra querymatrix of fixed length $k$ . The small query performs at-tention with the original context matrix, termed as packattention, to compress the context matrix to size $\mathbb { R } ^ { k \times d }$ .Subsequently, the regular attention, termed unpack atten-tion, applies attention to the original Q matrices and thecompressed K and V matrices. The extra query matrixcan be learnable parameters or acquired from previouslayers. Set Transformer [83] designs the similar techniqueby introducing an inducing points vector with fixed length.Unlike previous works that compress K and V, Funnel-Transformer [102] uses pooling operation to graduallycompress the sequence length of the Q matrix.

# 5.1.3 Transformer Alternates

In addition to applying efficient techniques to the attentionoperation, recent studies have also innovated to designsequence modeling architectures that are efficient yet ef-fective. Table 2 compares the efficiency of some represen-tative non-Transformer models. These architectures exhibitsub-quadratic computational complexity with respect to se-quence length during both training and inference, enablingLLMs to significantly increase their context length.

Within this research field, two prominent lines of studyhave garnered significant attention. One line of studies con-centrates on the State Space Model (SSM), which formulatessequence modeling as a recurrence transformation basedon the HiPPO theory [64]. Additionally, other studies pri-marily focus on employing long convolutions or designingattention-like formulations to model sequences.

State Space Model. The State Space Model (SSM) hasdemonstrated competitive modeling capabilities in certainNatural Language Processing (NLP) [75] and and ComputerVision (CV) [103] tasks. Compared to attention-based Trans-formers, SSM exhibits linear computational and memorycomplexity with respect to the input sequence length, whichenhances its efficiency in handling long-context sequences.In this survey, SSM refers to a series of model architecturesthat satisfy the following two properties: (1) They model

sequence based on the following formulation proposed byHiPPO [64] and LSSL [65]:

$$
x _ {k} = \underset {\overline {{\rightarrow}}} {\bar {A}} x _ {k - 1} + \bar {B} u _ {k}, \tag {7}
$$

$$
y _ {k} = \overline {{\boldsymbol {C}}} x _ {k},
$$

where A, $\overline { B }$ and $\overline { { C } }$ denote the transition matrices, $x$ denotesthe intermediate state and $u$ denotes the input sequence. (2)They design the transition matrix $A$ based on the HiPPOtheory [64]. Specifically, HiPPO proposes to compress theinput sequence into a sequence of coefficients (namely state)by projecting it onto a set of polynomial bases.

Building upon the aforementioned framework, severalstudies concentrate on improving the parameterization orinitialization of the transition matrix A. This involves re-fining how the matrix is formulated or initialized withinthe SSM to enhance its effectiveness and performance insequence modeling tasks. LSSL [65] firstly proposes to ini-tialize A with the optimal transition matrix HiPPO-LegSdesigned by HiPPO. In addition, LSSL also trains the SSMin a convolution manner by unrolling the Eq. 7. Specifically,through a convolution kernel defined as $\bar { \kappa } _ { L } ( A , B , C ) \ =$$( C A ^ { i } \bar { B } ) _ { i \in [ L ] } = ( C B , C A B , . . . , C A ^ { L - 1 } B )$ , the Eq. 7 canbe rewritten as $y ~ = ~ \mathcal { K } _ { L } ( \overline { { A } } , \overline { { B } } , \overline { { C } } ) \ast u$ and also can becomputed efficiently via Fast Fourier Transform (FFT). How-ever, computing this convolution kernel is expensive, sinceit requires multiple times of multiplication by $A$ . To thisend, S4 [66], DSS [67] and S4D [68] propose to diagonalizethe matrix $A _ { \cdot }$ , which can accelerate the computing. This canbe seen as a parameterization technique to the transitionmatrix A. Previous SSMs processed each input dimensionindependently, resulting in a large number of trainableparameters. To enhance efficiency, S5 [72] proposes to simul-taneously process all input dimensions using a single set ofparameters. Building upon this structure, S5 introduces aparameterization and initialization method for $A$ based onthe standard HiPPO matrix. Liquid S4 [71] and Mamba [75]parameterize the transition matrices in a input-dependentmanner, which further enhances the modeling capability ofSSM. Additionally, both S5 [72] and Mamba [75] adopt aparallel scan technique for efficient model training withoutthe need for convolution operations. This technique offersadvantages in implementation and deployment on modernGPU hardware.

Another line of research aim to design better modelarchitecture based on SSMs. GSS [69] and BiGS [74] com-bines the Gated Attention Unit (GAU) [104] with SSM.Specifically, they replace the attention operation in GAUwith SSM operation. BST [73] combines the SSM modelwith the proposed Block Transformer which introduces astrong local inductive bias. H3 [70] observes that SSM isweak in recalling the earlier tokens and comparing a tokenacross the sequence. To this end, it proposes to add a shiftSSM operation before the standard SSM operation, whichis used to directly shift the input tokens into the state.MambaFormer [76] combines the standard Transformer andSSM model by substituting the FFN layer in the Trans-former with an SSM layer. Jamba [105] introduces anotherapproach to combining the Transformer and SSM models byadding four Transformer layers into an SSM model. Dense-Mamba [106] explores the issue of hidden state degradation


TABLE 2Efficiency comparison of some novel non-Transformer models. Note that we denote $_ n$ as the input length and $d$ as the input dimension.


<table><tr><td rowspan="2">Model</td><td rowspan="2">Training Form</td><td rowspan="2">Training Computational Complexity</td><td rowspan="2">Training Memory Complexity</td><td rowspan="2">Inference Form</td><td colspan="2">Inference Computational Complexity</td></tr><tr><td>Prefilling</td><td>Decoding (per token)</td></tr><tr><td>Transformer [26]</td><td>Transformer-like</td><td>O(n2d)</td><td>O(n2+nd)</td><td>Transformer-like</td><td>O(n2d)</td><td>O(nd)</td></tr><tr><td>S4 [66]</td><td>Convolution</td><td>O(nd2 log n)</td><td>O(nd)</td><td>Recurrence</td><td>O(nd2)</td><td>O(d2)</td></tr><tr><td>Mamba [75]</td><td>Recurrence</td><td>O(nd2 log n)</td><td>O(nd)</td><td>Recurrence</td><td>O(nd2)</td><td>O(d2)</td></tr><tr><td>Hyena [61]</td><td>Convolution</td><td>O(nd log n)</td><td>O(nd)</td><td>Convolution</td><td>O(nd log n)</td><td>O(nd log n)</td></tr><tr><td>RetNet [63]</td><td>Transformer-like</td><td>O(n2d)</td><td>O(n2+nd)</td><td>Recurrence</td><td>O(nd2)</td><td>O(d2)</td></tr><tr><td>RWKV [62]</td><td>Recurrence</td><td>O(nd2)</td><td>O(nd)</td><td>Recurrence</td><td>O(nd2)</td><td>O(d2)</td></tr></table>

in traditional SSMs and introduces dense connections withinthe SSM architecture to preserve fine-grained informationacross deeper layers of the model. BlackMamba [107] andMoE-Mamba [108] propose to enhance SSM models with theMixture-of-Experts (MoE) technique to optimize the train-ing and inference efficiency while maintaining the modelperformance.

Other Alternates. In addition to SSMs, several other efficientalternates have also garnered significant attention, includinglong convolution and attention-like recurrence operation.

Several recent studies have applied long convolutionin the context of modeling long sequences [59], [60], [61].These investigations primarily concentrate on refining theparameterization of the convolution kernel. For instance,Hyena [61] employs an data-dependent parameterizationmethod for long convolutions using a shallow feed-forwardneural network (FFN).

Other studies [62], [63] aim to design the operation thathas a similar form as the attention operation but can beenrolled to the recurrent manner, enabling both efficienttraining and efficient inference. For instance, RWKV [62]builds upon AFT [109], which proposes to substitute theattention operation in the Transformer model with the fol-lowing equation:

$$
Y _ {t} = \sigma_ {q} \left(Q _ {t}\right) \odot \frac {\sum_ {t ^ {\prime} = 1} ^ {T} \exp \left(K _ {t ^ {\prime}} + w _ {t , t ^ {\prime}}\right) \odot V _ {t ^ {\prime}}}{\sum_ {t ^ {\prime} = 1} ^ {T} \exp \left(K _ {t ^ {\prime}} + w _ {t , t ^ {\prime}}\right)}, \tag {8}
$$

where $Q , K ,$ and $V$ are the query, key, and value matricesas in Transformer, $w \in \mathbb { R } ^ { T \times T }$ denotes a learnable pair-wise position bias and $\sigma _ { q } ( \cdot )$ denotes a non-linear function.Specifically, it further reparameterizes the position bias as$\begin{array} { r } { \dot { w } _ { t , t ^ { \prime } } = - ( t - t ^ { ' } ) w , } \end{array}$ and thus can rewrite Eq. 8 in a recursiveform. In this way, RWKV can combine the effective paral-lelizable training feature of Transformer and the efficientinference ability of RNN.

Efficiency Analysis. We analyze and compare the com-putational and memory complexity of several innovativeand representative non-transformer architectures in Table 2.In terms of training time, many studies (e.g., S4, Hyena,RetNet) aim to preserve training parallelism by adoptingtraining forms such as the convolution or attention. Notably,Mamba utilizes parallel scan techniques for processing inputsequences, thereby leveraging training parallelism as well.

On the other hand, during inference, most studies optfor recurrent architectures to maintain linear computationalcomplexity in the prefilling stage and to remain contextlength-agnostic in the decoding stage. Furthermore, in the

decoding phase, these novel architectures eliminate the needto cache and load features of previous tokens (similar to thekey-value cache in Transformer-based language models),resulting in significant memory access cost savings.

# 5.2 Model Compression

Model compression encompasses a range of techniques de-signed to enhance the inference efficiency of a pre-trainedmodel by modifying its data representation (e.g., quantiza-tion) or altering its architecture (e.g., sparsification, struc-tural optimization, and dynamic inference), as depicted inFig. 8.

# 5.2.1 Quantization

Quantization is a widely employed technique that reducesthe computational and memory cost of LLMs by convertingthe models’ weights and activations from high bit-width tolow bit-width representations. Specifically, many methodsinvolve quantizing FP16 tensors into low-bit integer tensors,which can be represented as follows:

$$
\mathbf {X} _ {\mathrm {I N T}} = \left[ \frac {\mathbf {X} _ {\mathrm {F P 1 6}} - Z}{S} \right], \tag {9}
$$

$$
S = \frac {\operatorname* {m a x} \left(\mathbf {X} _ {\mathrm {F P} 1 6}\right) - \operatorname* {m i n} \left(\mathbf {X} _ {\mathrm {F P} 1 6}\right)}{2 ^ {N - 1} - 1}, \tag {10}
$$

where $X _ { \mathrm { F P 1 6 } }$ denotes the 16-bit floating-point (FP16) value,$X _ { \mathrm { I N T } }$ denotes the low-precision integer value, $N$ denotesthe number of bits, and $S$ and $Z$ denote the scaling factorand zero-point.

In the following, we start with an efficiency analysis toillustrate how quantization techniques reduce the end-to-end inference latency of LLMs. Subsequently, we offer a de-tailed introduction to two distinct quantization workflows:Post-Training Quantization (PTQ) and Quantization-AwareTraining (QAT), respectively.

Efficiency Analysis. As discussed in Section 2.2, the infer-ence process of LLMs involves two stages: the prefillingstage and the decoding stage. During the prefilling stage,LLMs typically handle long token sequences, and the pri-mary operation is general matrix multiplication (GEMM).The latency of the prefilling stage is primarily constrainedby the computation performed by high-precision CUDACores. To address this challenge, existing methods quan-tize both weights and activations to accelerate computationusing low-precision Tensor Cores. As illustrated in Figure 9(b), activation quantization is performed online before eachGEMM operation, allowing computation with low-precision

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-11/924f71b2-d0c7-4bf4-8165-7633ffb6c93d/e2f35ca6bf44168f67cfaa0b5787978788bd5f53568710521b5083f65156e249.jpg)



Fig. 8. Taxonomy of model compression methods for Large Language Models.


Tensor Cores (e.g., INT8). This quantization approach isreferred to as Weight-Activation Quantization.

In contrast, during the decoding stage, LLMs processonly one token at each generation step using general matrix-vector multiplication (GEMV) as the core operation. Thelatency of the decoding stage is mainly influenced by theloading of large weight tensors. To tackle this challenge,existing methods focus on quantizing only the weights toaccelerate memory access. This method, known as Weight-only Quantization, involves offline quantization of weights,

followed by de-quantization of the low-precision weightsinto FP16 format for computation, as shown in Figure 9 (a).

Post-Training Quantization. Post-training quantization(PTQ) involves quantizing pre-trained models without theneed for retraining, which can be a costly process. WhilePTQ methods have been well-explored for smaller mod-els, applying existing quantization techniques directly toLLMs presents challenges. This is primarily because theweights and activations of LLMs often exhibit more outliersand have a wider distribution range compared to smaller


TABLE 3Summary of the representative studies on Post-Training Quantization. Quantized Tensor Type denotes which parts of tensors are quantized. DataFormat denotes whether to adopt uniform or non-uniform quantization. Quantization Parameter Determination Scheme denotes the how to decidethe parameters (e.g., scaling factor, zero-point). Quantized Value Update denotes whether to change the model weight (e.g., compensation,re-parameterization) during the quantization process.


<table><tr><td rowspan="2">Model</td><td colspan="3">Quantized Tensor Type</td><td rowspan="2">Data Format</td><td rowspan="2">Quantization Parameter Determination Scheme</td><td rowspan="2">Quantized Value Update</td></tr><tr><td>Weight</td><td>Activation</td><td>KV Cache</td></tr><tr><td>GPTQ [192]</td><td>✓</td><td></td><td></td><td>Uniform</td><td>Statistic-based</td><td>✓</td></tr><tr><td>LUT-GEMM [193]</td><td>✓</td><td></td><td></td><td>Non-uniform</td><td>Statistic-based</td><td></td></tr><tr><td>AWQ [194]</td><td>✓</td><td></td><td></td><td>Uniform</td><td>Search-based</td><td>✓</td></tr><tr><td>SqueezeLLM [197]</td><td>✓</td><td></td><td></td><td>Non-uniform</td><td>Statistic-based</td><td></td></tr><tr><td>LLM.int8() [204]</td><td>✓</td><td>✓</td><td></td><td>Uniform</td><td>Statistic-based</td><td></td></tr><tr><td>SmoothQuant [205]</td><td>✓</td><td>✓</td><td></td><td>Uniform</td><td>Statistic-based</td><td>✓</td></tr><tr><td>RPTQ [207]</td><td>✓</td><td>✓</td><td></td><td>Uniform</td><td>Statistic-based</td><td></td></tr><tr><td>OmniQuant [210]</td><td>✓</td><td>✓</td><td></td><td>Uniform</td><td>Search-based</td><td></td></tr><tr><td>FlexGen [203]</td><td>✓</td><td></td><td>✓</td><td>Uniform</td><td>Statistic-based</td><td></td></tr><tr><td>Atom [212]</td><td>✓</td><td>✓</td><td>✓</td><td>Uniform</td><td>Statistic-based</td><td></td></tr><tr><td>KVQuant [219]</td><td></td><td></td><td>✓</td><td>Non-uniform</td><td>Statistic-based</td><td></td></tr><tr><td>KIVI [220]</td><td></td><td></td><td>✓</td><td>Uniform</td><td>Statistic-based</td><td></td></tr></table>

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-11/924f71b2-d0c7-4bf4-8165-7633ffb6c93d/f4b66b0a9cdfdee00d5b9a8511f04e05d7437ea36efc1a431c3c1f5878c5a871.jpg)



(a) Weight-only Quantization


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-11/924f71b2-d0c7-4bf4-8165-7633ffb6c93d/b3075f86625e93eedb16d95508ab5152c99a9caf1607ffd5b05ee1e27e9a0be0.jpg)



(b) Weight-Activation Quantization



Fig. 9. (a) The inference workflow of Weight-only Quantization. (b) Theinference workflow of Weight-Activation Quantization.


models, making their quantization more challenging. Insummary, the complex nature of LLMs, characterized bytheir size and complexity, requires specialized approachesto effectively handle the quantization process. The presenceof outliers and wider distribution ranges in LLMs necessi-tates the development of tailored quantization techniquesthat can account for these unique characteristics withoutcompromising model performance or efficiency.

Numerous studies have concentrated on developingeffective quantization algorithms to compress LLMs. Weprovide a synthesis of representative algorithms categorizedacross four dimensions in Tab. 3. Regarding the types ofquantized tensors, certain studies [192], [193], [194], [197]concentrate on weight-only quantization, whereas manyothers [204], [205], [207] focus on quantizing both weightsand activations. Notably, in LLMs, the KV cache representsa distinctive component that impacts memory and memoryaccess. Consequently, some investigations [203], [212], [219]propose KV cache quantization. Regarding data formats,the majority of algorithms adopt a uniform format forstraightforward hardware implementation. Concerning the

determination of quantized parameters (e.g., scale, zero-point), most studies rely on statistics derived from weight oractivation values. Nevertheless, some research efforts [194],[210] advocate for searching optimal parameters based onreconstruction loss. Furthermore, certain studies [192], [194],[205] suggest updating unquantized weights (referred to asQuantized Value Update) before or during the quantizationprocess to enhance performance.

In weight-only quantization, GPTQ [192] represents anearly advancement in LLM quantization, building upon thetraditional algorithm OBQ [221]. OBQ utilizes an optimalquantization order per row of the weight matrix, guidedby the reconstruction error relative to the Hessian matrixof unquantized weights. After each quantization step, OBQiteratively adjusts the unquantized weights to mitigate re-construction errors. However, the frequent updating of theHessian matrix during quantization escalates computationalcomplexity. GPTQ streamlines this process by adopting auniform left-to-right order for quantizing each row, thus cir-cumventing the need for extensive Hessian matrix updates.This strategy substantially reduces computational demandsby computing the Hessian matrix solely during the quanti-zation of one row, then leveraging the computing resultsfor subsequent rows, expediting the overall quantizationprocedure. LUT-GEMM [193] presents a novel dequantiza-tion method utilizing a Look-Up Table (LUT), aiming toaccelerate the inference process of quantized LLMs by re-ducing the dequantization overhead. Additionally, it adoptsa non-uniform quantization approach known as Binary-Coding Quantization (BCQ), which incorporates learnablequantization intervals. AWQ [194] observes that weightchannels vary in importance for performance, particularlyemphasizing those aligned with input channels exhibitingoutliers in activations. To enhance the preservation of crit-ical weight channels, AWQ utilizes a reparameterizationmethod. This technique selects reparameterization coeffi-cients via grid search to minimize reconstruction errorseffectively. OWQ [195] observes the difficulty of quantiz-ing weights associated with activation outliers. To addressthis challenge, OWQ employs a mixed-precision quanti-zation strategy. This method identifies weak columns in

the weight matrix and allocates higher precision to thesespecific weights, while quantizing the rest of the weights at alower precision level. SpQR [196] introduces a methodologywhere weight outliers are identified and allocated higherprecision during quantization, while the rest of the weightsare quantized to 3 bits. SqueezeLLM [197] proposes to storethe outliers in a full-precision sparse matrix, and applynon-uniform quantization to the remaining weights. Thevalues for non-uniform quantization are determined basedon quantization sensitivity, which contributes to improvedperformance of the quantized model. QuIP [198] introducesLDLQ, an optimal adaptive method for a quadratic proxyobjective. The study reveals that ensuring incoherence be-tween weight and Hessian matrices can enhance the ef-fectiveness of LDLQ. QuIP utilizes LDLQ and achievesincoherence by employing random orthogonal matrix mul-tiplication. FineQuant [199] utilizes a heuristic approachto determine the granularity of quantization per column,combining empirical insights gained from experiments todesign a quantization scheme. QuantEase [200] builds uponGPTQ. When quantizing each layer, it proposes a methodbased on Coordinate Descent to compensate for the un-quantized weights more precisely. Additionally, QuantEasecan leverage quantized weights from GPTQ as an initial-ization and further refine the compensation process. LLM-MQ [201] protects the weight outliers with FP16 format,and stores them in Compressed Sparse Row (CSR) formatfor efficient computation. Besides, LLM-MQ models the bit-width assignment to each layer as an integer programmingproblem, and employs an efficient solver to solve it within afew seconds. Moveover, LLM-MQ designs a efficient CUDAkernel to integrate dequantization operators, thereby reduc-ing memory access cost during computation. Inspired bythe equivalent transformations used in the previous PTQmethods, AffineQuant [215] firstly introduces equivalentaffine transformations in quantization, which extends theoptimization scope and further reduces the quantization er-rors. Recently, many studies [198], [216], [217], [218] followsthe computational invariance idea, by multiplying rotationmatrices to the weight matrices and activation matrices. Inthis way, they can effectively eliminate the outliers in theweights and activations, thus help to quantize the LLMs.These studies use different rotation matrices. For example,QuaRot [217] applies randomize Hadamard transformationsto the weights and activations. SpinQuant [218] finds theoptimal rotation matrices by training on a small validationdataset.

For weight-activation quantization, ZeroQuant [202] em-ploys finer-grained quantization for weights and activa-tions, leveraging kernel fusion to minimize the memoryaccess cost during quantization and conducting layer-by-layer knowledge distillation to recover the performance.FlexGen [203] quantizes weights and KV cache directlyinto INT4 to reduce the memory footprint during infer-ence with large batch sizes. LLM.int8() [204] identifies thatoutliers in activations are concentrated within a small sub-set of channels. Leveraging this insight, LLM.int8() splitsactivations and weights into two distinct parts based onthe outlier distribution within input channels to minimizequantization errors in activations. Channels containing out-lier data in both activations and weights are stored in

FP16 format, while other channels are stored in INT8format. SmoothQuant [205] employs a reparameterizationtechnique to address the challenges of quantizing activa-tion values. This method introduces a scaling factor thatexpands the data range of weight channels while shrinkingthe data range of corresponding activation channels. Zero-Quant [202] introduces a group-wise quantization strategyfor weights and a token-wise quantization approach foractivations. Building upon this methodology, ZeroQuant-V2 [206] presents the LoRC (Low-Rank Compensation) tech-nique, employing low-rank matrices to mitigate quantiza-tion inaccuracies. RPTQ [207] identifies substantial varia-tions in the distribution of different activation channels,which present challenges for quantization. To mitigate thisissue, RPTQ reorganizes channels with similar activationdistributions into clusters and independently applies quan-tization within each cluster. OliVe [208] observes that thenormal values neighboring to the outliers are less critical.Therefore, it pairs each outlier with a normal value, sacri-ficing the latter to achieve a broader representation rangefor outliers. ${ \mathrm { O S + } }$ [169] observes that the distribution of out-liers is concentrated and asymmetrical, posing a challengeto LLM quantization. To address this, ${ \mathrm { O S + } }$ introduces achannel-wise shifting and scaling technique aimed at allevi-ating these challenges. The shifting and scaling parametersare determined through a search process to effectively han-dle the concentrated and asymmetrical outlier distribution.ZeroQuant-FP [209] investigates the feasibility of quantizingweight and activation values into FP4 and FP8 formats.The study reveals that quantizing activations into floating-point types (FP4 and FP8) produces superior results com-pared to integer types. Omniquant [210] diverges from priorapproaches that rely on empirical design of quantizationparameters. Instead, it optimizes the boundaries for weightclipping and the scaling factor for equivalent transformationto minimize quantization errors. QLLM [211] addressesthe impact of outliers on quantization by implementingchannel reassembly. Additionally, it introduces learnablelow-rank parameters to minimize quantization errors inthe post-quantized model. Atom [212] employs a strategyinvolving mixed-precision and dynamic quantization foractivations. Notably, it extends this approach to quantize theKV cache into INT4 to enhance throughput performance.LLM-FP4 [187] endeavors to quantize the entire modelinto FP4 format and introduces a pre-shifted exponentbias technique. This approach combines the scaling factorof activation values with weights to address quantizationchallenges posed by outliers. BiLLM [213] represents oneof the lowest-bit PTQ efforts to date. BiLLM identified thebell-shaped distribution of weights and the exceptionallylong-tail distribution of weights’ Hessian matrix. Based onthis, it proposes to categorize weights into salient and non-salient values structurally based on the Hessian matrixand binarizes them separately. As a result, BiLLM canextensively quantize LLMs to 1.08 bits without significantdegradation in perplexity. KVQuant [219] proposes a non-uniform quantization scheme for KV cache quantization, byderiving the optimal datatypes offline on a calibration set.KIVI [220] proposes a tuning-free 2bit KV cache quantiza-tion algorithm, which utilizes per-channel quantization forkey cache and per-token quantization for value cache in a


TABLE 4Comparison of speed-ups in different scenarios (e.g., model size, batch size, input context length, inference framework) with W4A16 quantizationbased on TensorRT-LLM [222] and LMDeploy [223] framework, respectively. We test the speed-ups of prefilling/decoding/end-to-end latency on asingle NVIDIA A100 GPU. OOM denotes “Out Of Memory”.


<table><tr><td rowspan="2"></td><td rowspan="2">B</td><td colspan="5">TensorRT-LLM</td></tr><tr><td>128</td><td>256</td><td>512</td><td>1024</td><td>2048</td></tr><tr><td rowspan="5">LLaMA-2-7B</td><td>1</td><td>1.06/2.40/2.37</td><td>0.90/2.38/2.34</td><td>0.92/2.30/2.28</td><td>0.88/2.19/2.17</td><td>0.91/2.00/1.98</td></tr><tr><td>2</td><td>0.88/2.10/2.05</td><td>0.91/2.07/2.04</td><td>0.89/2.01/1.98</td><td>0.91/1.92/1.89</td><td>0.88/1.78/1.76</td></tr><tr><td>4</td><td>0.92/1.72/1.67</td><td>0.89/1.67/1.64</td><td>0.90/1.61/1.58</td><td>0.87/1.53/1.51</td><td>0.84/1.42/1.40</td></tr><tr><td>8</td><td>0.91/1.43/1.36</td><td>0.88/1.38/1.33</td><td>0.83/1.33/1.28</td><td>0.77/1.25/1.21</td><td>0.78/1.16/1.14</td></tr><tr><td>16</td><td>0.91/1.43/1.36</td><td>0.88/1.38/1.33</td><td>0.83/1.33/1.28</td><td>0.77/1.25/1.21</td><td>0.78/1.16/1.14</td></tr><tr><td></td><td>B</td><td>128</td><td>256</td><td>512</td><td>1024</td><td>2048</td></tr><tr><td rowspan="5">LLaMA-2-13B</td><td>1</td><td>1.24/2.51/2.50</td><td>0.89/2.45/2.47</td><td>0.94/2.34/2.42</td><td>0.90/2.18/2.32</td><td>0.83/1.94/2.16</td></tr><tr><td>2</td><td>0.90/2.51/2.50</td><td>0.95/2.45/2.47</td><td>0.90/2.34/2.42</td><td>0.83/2.18/2.32</td><td>0.80/1.94/2.16</td></tr><tr><td>4</td><td>0.96/1.80/1.76</td><td>0.91/1.78/1.74</td><td>0.83/1.73/1.69</td><td>0.80/1.65/1.62</td><td>0.83/1.54/1.52</td></tr><tr><td>8</td><td>0.91/1.86/1.77</td><td>0.83/1.81/1.73</td><td>0.80/1.73/1.66</td><td>0.82/1.62/1.56</td><td>0.75/1.46/1.41</td></tr><tr><td>16</td><td>0.84/1.84/1.69</td><td>0.81/1.77/1.63</td><td>0.82/1.63/1.53</td><td>0.78/1.46/1.39</td><td>OOM</td></tr><tr><td rowspan="2"></td><td rowspan="2">B</td><td colspan="5">LMDeploy</td></tr><tr><td>128</td><td>256</td><td>512</td><td>1024</td><td>2048</td></tr><tr><td rowspan="5">LLaMA-2-7B</td><td>1</td><td>1.30/2.11/2.09</td><td>0.94/2.07/2.05</td><td>0.90/2.03/2.02</td><td>0.88/1.97/1.96</td><td>0.94/1.92/1.91</td></tr><tr><td>2</td><td>1.03/2.24/2.20</td><td>0.90/2.19/2.15</td><td>0.88/2.11/2.08</td><td>0.93/1.97/1.95</td><td>0.85/1.78/1.76</td></tr><tr><td>4</td><td>0.90/2.18/2.10</td><td>0.87/2.12/2.05</td><td>0.93/2.01/1.96</td><td>0.92/1.86/1.83</td><td>0.92/1.64/1.62</td></tr><tr><td>8</td><td>0.92/1.92/1.77</td><td>0.91/1.82/1.71</td><td>0.92/1.65/1.57</td><td>0.93/1.45/1.41</td><td>0.94/1.28/1.26</td></tr><tr><td>16</td><td>0.92/1.92/1.77</td><td>0.91/1.82/1.71</td><td>0.92/1.65/1.57</td><td>0.93/1.45/1.41</td><td>0.94/1.28/1.26</td></tr><tr><td></td><td>B</td><td>128</td><td>256</td><td>512</td><td>1024</td><td>2048</td></tr><tr><td rowspan="5">LLaMA-2-13B</td><td>1</td><td>1.32/2.34/2.32</td><td>0.94/2.31/2.28</td><td>0.92/2.22/2.20</td><td>0.94/2.15/2.13</td><td>0.94/2.01/1.99</td></tr><tr><td>2</td><td>1.06/2.42/2.36</td><td>0.92/2.37/2.32</td><td>0.94/2.29/2.25</td><td>0.94/2.15/2.12</td><td>0.95/1.95/1.93</td></tr><tr><td>4</td><td>0.93/2.36/2.26</td><td>0.94/2.29/2.21</td><td>0.94/2.18/2.12</td><td>0.95/2.01/1.97</td><td>0.96/1.78/1.75</td></tr><tr><td>8</td><td>0.92/2.24/2.10</td><td>0.93/1.93/2.02</td><td>0.94/1.81/1.89</td><td>0.94/1.65/1.71</td><td>0.95/1.45/1.49</td></tr><tr><td>16</td><td>0.93/2.02/1.85</td><td>0.94/1.90/1.76</td><td>0.94/1.73/1.63</td><td>0.95/1.50/1.45</td><td>OOM</td></tr></table>

group-wise manner. Li et al. [214] conducted a thoroughevaluation to assess the impact of quantization on differenttensor types (including KV Cache), various tasks, 11 LLMfamilies, and SOTA quantization methods.

Quantization-Aware Training. Quantization-aware training(QAT) incorporates the influence of quantization within themodel training procedure. By integrating layers that repli-cate quantization effects, this approach facilitates weightadaptation to quantization-induced errors, leading to en-hanced task performance. Nevertheless, training LLMs typ-ically demands substantial training data and considerablecomputational resources, posing potential bottlenecks forQAT implementation. Consequently, current research en-deavors focus on strategies to reduce the training data re-quirements or alleviate the computational burden associatedwith QAT implementation.

To reduce the data requirements, LLM-QAT [187] intro-duces a data-free method to generate the training data byusing the original FP16 LLMs. Specifically, LLM-QAT usesevery token in the tokenization vocabulary as a startingtoken to generate sentences. Based on the generated trainingdata, LLM-QAT applies a distillation-based workflow totrain the quantized LLM to match the output distributionof the original FP16 LLM. Norm Tweaking [188] limitsthe selection of the starting token to only those languagecategories listed among the top languages with the highestproportion. This strategy can effectively improve the gener-alization of the quantized model on different tasks.

To reduce the computation cost, many methods applyparameter-efficient tuning (PEFT) strategies to accelerate

QAT. QLoRA [189] quantizes the weights of LLMs into4-bit and subsequently employs LoRA [224] in BF16 foreach 4-bit weight matrix to fine-tune the quantized model.QLoRA allows for the efficient fine-tuning of a 65B param-eter LLM on one GPU with only 30GB of memory. QA-LoRA [190] proposes to incorporate group-wise quantiza-tion into QLoRA. The authors observe that the number ofquantization parameters in QLoRA is significantly smallerthan the number of LoRA parameters, leading to an imbal-ance between quantization and low-rank adaptation. Theysuggest that group-wise operations can address this issue byincreasing the number of parameters dedicated to quantiza-tion. In addition, QA-LoRA can merge the LoRA terms intothe corresponding quantized weight matrices. LoftQ [191]identifies that initializing LoRA matrices with zeros inQLoRA is inefficient for downstream tasks. As an alterna-tive, LoftQ suggests initializing the LoRA matrices usingthe Singular Value Decomposition (SVD) of the differencebetween the original FP16 weights and quantized weights.LoftQ iteratively applies quantization and SVD to achieve amore accurate approximation of the original weights. NormTweaking [188] proposes to train the LayerNorm layer afterquantization and use knowledge distillation to match theoutput distribution of the quantized model with that of theFP16 model, achieving effects similar to LLM-QAT whileavoiding high training costs.

Comparative Experiments and Analysis. In this section, weconduct experiments to evaluate the speed-ups achievedby employing the weight-only quantization technique invarious scenarios. Specifically, we focus on two widely-used

large language models (LLMs), LLaMA-2-7B and LLaMA-2-13B, and quantize their weights to 4-bit using the AWQ [194]algorithm. Subsequently, we deploy these quantized modelson a single NVIDIA A100 GPU using two different inferenceframeworks: TensorRT-LLM [222] and LMDeploy [223]. Wethen evaluate the speed-ups achieved by these frameworksacross different input sequences characterized by varyingbatch sizes and context lengths.

We present the speed-ups of prefilling latency, decodinglatency, and end-to-end latency, as summarized in Tab. 4.From the results, several key observations can be made:(1) Weight-only quantization can substantially accelerate thedecoding stage, leading to improvements in end-to-end la-tency. This enhancement primarily stems from the capabilityof loading the quantized model with low-precision weighttensors much more swiftly from the High Bandwidth Mem-ory (HBM), as illustrated in the preceding “Efficient Analy-sis” part. Consequently, this approach markedly diminishesthe memory access overhead. (2) Regarding the prefillingstage, weight-only quantization may actually increase thelatency. This is due to the fact that the bottleneck in theprefilling stage is the computational cost rather than thememory access cost. Therefore, quantizing only the weightswithout the activations has minimal impact on latency. Ad-ditionally, as illustrated in Fig. 9, weight-only quantizationnecessitates the de-quantization of low-precision weightsto FP16, leading to additional computational overhead andconsequently slowing down the prefilling stage. (3) As thebatch size and input length increase, the extent of speed-upachieved by weight-only quantization gradually diminishes.This is primarily because, with larger batch sizes and inputlengths, the computational cost constitutes a larger propor-tion of latency. While weight-only quantization predomi-nantly reduces memory access cost, its impact on latencybecomes less significant as the computational demandsbecome more prominent with larger batch sizes and inputlengths. (4) Weight-only quantization offers greater benefitsfor larger models due to the significant memory access over-head associated with larger model sizes. As models growin complexity and size, the amount of memory requiredto store and access weights increases proportionally. Byquantizing the model weights, weight-only quantization ef-fectively reduces this memory footprint and memory accessoverhead.

# 5.2.2 Sparsification

Sparsification is a compression technique that increases theproportion of zero-valued elements in data structures suchas model parameters or activations. This method aims todecrease computational complexity and memory usage byefficiently ignoring zero elements during computation. Inthe context of LLMs, sparsification is commonly appliedto weight parameters and attention activations. It leads tothe development of weight pruning strategies and sparseattention mechanisms.

Weight Pruning. Weight pruning systematically removesless critical weights and structures from models, aimingto reduce computational and memory cost during bothprefilling stages and decoding stages without significantlycompromising performance. This sparsification approach iscategorized into two main types: unstructured pruning and

structured pruning. The categorization is based on the gran-ularity of the pruning process, as illustrated in Figure 10.

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-11/924f71b2-d0c7-4bf4-8165-7633ffb6c93d/7ed9f83dea943ff6e9e85a67a47bb87f699a8b650da0cb3260e4a020d16570cd.jpg)



Unstructured PruningGranularity: Weight


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-11/924f71b2-d0c7-4bf4-8165-7633ffb6c93d/ccffe4c41a736eb87e24213db10ccaa3986a9d7b7b4da90bb7d20a0d59298a0a.jpg)



Structured PruningGranularity: Channel/Group/Layer


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-11/924f71b2-d0c7-4bf4-8165-7633ffb6c93d/291af509613c7bbd38e1078f2da6082e5d0e8fffddffc73863ba8baf21601972.jpg)



Fig. 10. Illustration of Unstructured Pruning (left) and Structured Pruning(right).


Unstructured pruning prunes individual weight valueswith fine granularity. Compared with structured pruning, ittypically achieves a greater level of sparsity with minimalimpact on model prediction. However, the sparse patternachieved through unstructured pruning lacks high-levelregularity, leading to irregular memory access and compu-tation patterns. This irregularity can significantly hinder thepotential for hardware acceleration, as modern computingarchitectures are optimized for dense, regular data patterns.Consequently, despite achieving higher sparsity levels, thepractical benefits of unstructured pruning in terms of hard-ware efficiency and computational speedup may be limited.

The common focus of this line of work is the pruningcriterion, including the weight importance and pruningratio. Considering the huge parameter size of LLMs, im-proving the pruning efficiency is also crucial. One pruningcriterion is to minimize the reconstruction loss of the model.SparseGPT [165] is a representative approach in this field. Itfollows the idea of OBS [225], which considers the impact ofremoving each weight on the network’s reconstruction loss.OBS iteratively decides a pruning mask to prune the weightsand reconstructs the unpruned weights to compensate forthe pruning loss. SparseGPT overcomes the efficiency bot-tleneck of OBS via the Optimal Partial Updates technique,and designs an adaptive mask selection technique basedon the OBS reconstruction error. Prune and Tune [168]improves upon SparseGPT by fine-tuning the LLMs withminimal training steps during pruning. ISC [167] designs anovel pruning criterion by combining the saliency criteriain OBS [225] and OBD [226]. It further assigns non-uniformpruning ratios to each layer based on Hessian information.oBERT [171] and FastPruning [172] utilizes the second-order information of the loss function to decide the prunedweights. BESA [170] learns a differentiable binary mask viagradient descent of the reconstruction loss. The pruning ra-tio for each layer is sequentially decided by minimizing thereconstruction error. The other popular pruning criterion ismagnitude-based. Wanda [166] proposes to use the element-wise product between the weight magnitude and the normof input activation as the pruning criterion. RIA [173] jointlyconsiders the weights and activations by using the metric ofRelative Importance and Activations, which evaluates theimportance of each weight element based on all its con-nected weights. In addition, RIA converts the unstructuredsparsity pattern to a structured N:M sparsity pattern, whichcan enjoy the actual speed-up on NVIDIA GPUs. The recentstudy, Pruner-Zero [185], proposes to automatically identify

the optimal pruning metric for LLMs, going beyond thehand-designed matrics. As a result, the optimal metric tai-lored for LLaMA and LLaMA-2 is $W { \odot } W { \odot } \sigma ( G )$ , where Wand $\pmb { G }$ represent the weights and gradients, and $\sigma ( \cdot )$ scalesa tensor to [0,1] using its mininum and maximum value.Additionally, OWL [169] focuses on deciding the pruningratio of each layer. It assigns the pruning ratios to each layerbased on its activation outlier ratios. DSØT [186] proposesa training-free appoarch to fine-tune the pruned LLMs. Itbuilds upon the “pruning-and-growing” workflow adoptedin Dynamic Sparse Training [227], which first prunes themodel and then iteratively adjusts the network topologywithout training or weight update. DSØT further designsthe pruning and growing metrics tailored for LLMs.

Structured pruning prunes larger structural units ofthe model, such as entire channels or layers, operating ata coarser granularity compared to unstructured pruning.These methods directly facilitate inference speed-up on con-ventional hardware platforms due to their alignment withthe dense, regular data patterns these systems are optimizedto process. However, the coarse granularity of structuredpruning often results in a more pronounced impact onmodel performance. The pruning criterion of this line ofwork additionally enforces the structured pruning pattern.LLM-Pruner [174] proposes a task-agnostic structured prun-ing algorithm. Specifically, it first identifies the couple struc-tures in the LLM, based on the connection dependenciesbetween neurons. Then, it decides which structure groupsto remove based on a well-designed group-wise pruningmetric. After pruning, it further proposes to recover themodel performance by a parameter-efficient training tech-nique, i.e., LoRA [224]. Sheared LLaMA [175] proposes toprune the original LLM to a specific target architecture ofexisting pre-trained LLMs. In addition, it designs dynamicbatch-loading techniques to improve post-training perfor-mance. ZipLM [176] iteratively identifies and prunes thestructural components with the worst trade-off betweenloss and runtime. LoRAPrune [177] proposes a structuredpruning framework for the pre-trained LLMs with LoRAmodules to enable fast inference of LoRA-based models.It designs a LoRA-guided pruning criterion that uses theweights and gradients of LoRA, and an iterative pruningscheme to remove the unimportant weights based on thecriterion. LoRAShear [178] also designs a pruning methodfor LoRA-based LLMs with (1) a graph algorithm to identifythe minimal removal structures, (2) a progressive structuredpruning algorithm LHSPG, and (3) a dynamic knowledgerecovery mechanism to recover the model performance.SliceGPT [179] builds on the idea of computational invari-ance of RMSNorm operation. It proposes to structurallyarrange the sparsity in each weight matrix, and to sliceout the entire rows or columns. PLATON [180] proposesto prune the weights by considering both their importanceand uncertainty. It uses the exponential moving average(EMA) of the importance scores to estimate the importance,and adopts the upper confidence bound (UCB) for theuncertainty. CoFi [181] and SIMPLE [182] propose to prunethe attention head, FFN neurons and hidden dimension vialearning the corresponding sparsity masks. After pruning,they further adopt knowledge distillation to fine-tune thepruned models for performance recovery. MoE techniques

(Sec. 5.1.1) have attracted much attention in the field ofefficient LLMs. Recent studies tend to explore the expertpruning methods for MoE-based LLMs. For example, Ex-pertSparsity [183] proposes to prune some less importantFFN experts in each model layer. Specifically, it utilizesthe Frobenius norm of the difference between the originaloutput and the output of the pruned layer to quantify theloss of pruned experts. In constrast, SEER-MoE [184] usesthe total number of times that one expert gets activated ona calibration dataset, to quantify this expert’s importance.


local global random


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-11/924f71b2-d0c7-4bf4-8165-7633ffb6c93d/68c0211f6051753b77f8908d2465a6702f0447267e09301fabd785c753ea38dc.jpg)



(a)



dilated rate 1/2/8


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-11/924f71b2-d0c7-4bf4-8165-7633ffb6c93d/4e176f35aea90ea6c714b2e69cd1fb98f3962f986a5f7475a22f6a827e8c9054.jpg)



(b)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-11/924f71b2-d0c7-4bf4-8165-7633ffb6c93d/cdefe189e49dc5f91d748251786c5f04f5a6b04a751f166c41ee0f192a1caadd.jpg)



(c)



bucket 0/1


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-11/924f71b2-d0c7-4bf4-8165-7633ffb6c93d/04d14c3405fc6987d8399eb3d11041ea4ba36029603956215009e08c76f06ac2.jpg)



Fig. 11. Examples of different sparse attention masks. (a) Static maskwith local, global, and random attention pattern. (b) Static mask withdilated attention pattern of different dilated rate. (c) Dynamic tokenpruning. (d) Dynamic attention pruning.


Sparse Attention. Sparse attention techniques in Multi-Head Self-Attention (MHSA) components of transformermodels strategically omit certain attention calculations toenhance computational efficiency of the attention operationmainly in the prefilling stage. These mechanisms divergeinto static and dynamic categories based on their relianceon specific input data.

Static sparse attention removes activation values inde-pendently of specific inputs [150], [152], [153], [154]. Thesemethods pre-determine the sparse attention mask and en-force it on the attention matrix during inference. Previousstudies combine different sparse patterns to preserve themost essential elements within each attention matrix. Asshown in Figure 11(a), the most common sparse attentionpatterns are the local and global attention patterns. The localattention pattern captures the local context of each tokenwith a fixed-size window attention surrounding each token.The global attention pattern captures the correlation of spe-cific tokens to all other tokens by computing and attendingto all tokens across the sequence. Note that leveraging globalpatterns can eliminate the need to store key-value (KV)pairs for unused tokens, thereby reducing memory accesscost and memory usage during the decoding stage. SparseTransformer [150] combines these patterns to capture the

local context with a local pattern, and then aggregates theinformation with the global pattern for every few words.StreamingLLM [151] applies the local pattern, along withthe global pattern only for the first few tokens. It showsthat such a global pattern serves as the attention sink tokeep the strong attention scores toward initial tokens. Ithelps the LLMs to generalize to infinite input sequencelength. Bigbird [153] also uses the random pattern, whereall tokens attend to a set of random tokens. The combi-nation of local, global and random patterns is proven toencapsulate all continuous sequence-to-sequence functions,affirming its Turing completeness. As shown in Figure 11(b),Longformer [152] additionally introduces the dilated slidingwindow pattern. It is analogous to dilated CNNs and makesthe sliding window “dilated” to increase the receptivefield. To adapt the model to the sparse setting, StructuredSparse Attention [154] advocates an entropy-aware trainingmethod that congregates high-probability attention valuesinto denser regions. Unlike previous studies that manuallydesign sparse patterns, SemSA [155] uses gradient-basedprofiling to identify important attention patterns and au-tomatically optimizes the attention density distribution tofurther improve model efficiency.

In contrast, Dynamic sparse attention adaptively elim-inates activation values based on varying inputs, employ-ing real-time monitoring of neuronal activation values tobypass computations for neurons with negligible impact,thereby achieving pruning. Most dynamic sparse attentionmethods employ the dynamic token-pruning methods, asFigure 11(c) shows. Spatten [156], SeqBoat [157] and Adap-tively Sparse Attention [158] leverage the inherent redun-dancy in linguistic constructs to propose dynamic token-level pruning strategies. Spatten [156] assesses the cumula-tive importance of each word by aggregating the attentionmatrix columns, subsequently pruning tokens with minimalcumulative significance from the input in subsequent layers.SeqBoat [157] trains a linear State Space Model (SSM) with asparse sigmoid function to determine which token to prunefor each attention head. Both Spatten and SeqBoat prunethe uninformative tokens for the whole input. AdaptivelySparse Attention [158] gradually prunes the tokens duringthe generation process. It drops parts of the context that areno longer required for future generation.

In addition to dynamic token pruning, dynamic atten-tion pruning strategies are also employed [159], [160], [161],[162], [163]. As Figure 11(d) shows, instead of pruning allthe attention values of certain tokens, these methods dy-namically prune the selective part of the attention based onthe input. A prominent approach within this domain is dy-namically segmenting input tokens into groups, known asbuckets, and strategically omitting the attention calculationsfor tokens that reside in separate buckets. The challengeand focus of these methods lie in the way to cluster relatedtokens together, thereby facilitating attention computationssolely among them to enhance efficiency. Reformer [159]leverages locality-sensitive hashing to cluster keys andqueries that share identical hash codes into the same bucket.Following this, Sparse Flash Attention [160] introduces spe-cialized GPU kernels optimized for this hash-based sparseattention mechanism, further improving computational effi-ciency. Meanwhile, the Routing Transformer [161] employs a

spherical k-means clustering algorithm to aggregate tokensinto buckets, optimizing the selection process for attentioncomputations. Sparse Sinkhorn Attention [162] adopts alearned sorting network to align keys with their relevantquery buckets, ensuring that attention is computed onlybetween the corresponding query-key pairs. Diverging fromthe bucket-level operation, $_ \mathrm { H _ { 2 } O }$ [163] introduces the token-level dynamic attention pruning mechanism. It combinesstatic local attention with dynamic computations betweenthe current query and a set of dynamically identified keytokens, termed heavy-hitters $\mathrm { ( H _ { 2 } ) }$ . These heavy-hitters aredynamically adjusted with an eviction policy aimed at re-moving the least significant keys at each generation step,effectively managing the size and relevance of the heavy-hitter set.

Moreover, viewing each token as a graph node andattention between tokens as edges offers an extended per-spective on static sparse attention [153], [164]. The original,full attention mechanism equates to a complete graph witha uniform shortest path distance of 1. Sparse attention,with its random mask, introduces random edges, effectivelyreducing the shortest path distance between any two nodesto $O ( \log n )$ , thus maintaining efficient information flowakin to full attention. Diffuser [164] utilizes the perspectiveof graph theory to expand the receptive field of sparseattention with multi-hop token correlations. It also takesinspiration from the expander graph properties to designbetter sparse patterns that approximate the information flowof full attention.

Beyond the attention-level and token-level sparsity, thescope of attention pruning extends to various granularities.Spatten [156] also extends pruning beyond token granular-ity to attention head granularity, eliminating computationsfor inessential attention heads to further reduce computa-tional and memory demands.

# 5.2.3 Structure Optimization

The objective of structure optimization is to refine modelarchitecture or structure with the goal of enhancing thebalance between model efficiency and performance. Withinthis field of research, two prominent techniques stand out:Neural Architecture Search (NAS) and Low Rank Factoriza-tion (LRF).

Neural Architecture Search. Neural Architecture Search(NAS) [228] aims to automatically search the optimal neu-ral architectures that strike an optimized balance betweenefficiency and performance. AutoTinyBERT [138] utilizesone-shot Neural Architecture Search (NAS) to discover thehyper-parameters of the Transformer architecture. Notably,it introduces a compelling batch-wise training approachto train a Super Pre-trained Language Model (SuperPLM)and subsequently employs an evolutionary algorithm toidentify the optimal sub-models. NAS-BERT [139] trains alarge super-net on conventional self-supervised pre-trainingtasks using several innovative techniques, such as block-wise search, search space pruning, and performance ap-proximation. This approach allows NAS-BERT to be appliedefficiently across various downstream tasks without requir-ing extensive re-training. Structure pruning via NAS [140]treats structural pruning as a multi-objective NAS problem,

and solves it via the one-shot NAS method. LiteTransform-erSearch [141] proposes to use a training-free indicator, i.e.,the number of parameters, as a proxy indicator to guide thesearch. This method enables efficient exploration and selec-tion of the optimal architectures without the need for actualtraining during the search phase. AutoDistil [142] presents afully task-agnostic few-shot NAS algorithm featuring threeprimary techniques: search space partitioning, task-agnosticSuperLM training, and task-agnostic search. This approachaims to facilitate efficient architecture discovery across vari-ous tasks with minimal task-specific adaptations. Typically,NAS algorithms necessitate evaluating the performance ofeach sampled architecture, which can incur significant train-ing cost. Consequently, these techniques are challenging toapply to LLMs.

Low Rank Factorization. Low Rank Factorization (LRF), orLow Rank Decomposition, aims to approximate a matrix$A ^ { m \times n }$ with two low-rank matrices $B ^ { m \times r }$ and $C ^ { r \times n }$ by:

$$
A ^ {m \times n} \approx B ^ {m \times r} \times C ^ {r \times n}, \tag {11}
$$

where $r$ is much smaller than $m$ and $n$ . In this way, LRFcan diminish memory usage and enhance computationalefficiency. Furthermore, during the decoding stage of LLMinference, memory access cost presents a bottleneck to thedecoding speed. Therefore, LRF can reduce the numberof parameters that need to be loaded, thereby accelerat-ing the decoding speed. LoRD [143] shows the potentialof compressing the LLMs without largely degrading theperformance via LRF. Specifically, it adopts Singular ValueDecomposition (SVD) to factorize the weight matrices, andsuccessfully compresses a LLM with 16B parameters to12.3B with minimal performance drop. TensorGPT [144]introduces a method to compress the embedding layerusing Tensor-Train Decomposition. Each token embeddingis treated as a Matrix Product State (MPS) and efficientlycomputed in a distributed manner. LoSparse [145] combinesthe benefits of LRF and weight pruning for LLM com-pression. By leveraging low-rank approximation, LoSparsemitigates the risk of losing too many expressive neurons thattypically occurs with direct model pruning. LPLR [146] andZeroQuant-V2 [147] both propose to compress the weightmatrix by simultaneously applying LRF and quantization toit. DSFormer [148] proposes to factorize the weight matrixinto the product of a semi-structured sparse matrix anda small dense matrix. ASVD [149] designs an activation-aware SVD method. This approach involves scaling theweight matrix based on activation distribution prior to ap-plying SVD for matrix decomposition. ASVD also involvesdetermining an appropriate truncation rank for each layerthrough a search process. SVD-LLM [229] analyses the re-lationship between the singular values of the transformedweight matrices and the compression loss. Then, it designsa truncation-aware data whitening technique to identify thesingular value that causes the minimal loss after removingit. Additionally, SVD-LLM develops a layer-wise closed-form update strategy to recover the task performance afterthe factorization.

# 5.2.4 Knowledge Distillation

Knowledge Distillation (KD) is a well-established techniquefor model compression, wherein knowledge from large

models (referred to as teacher models) is transferred tosmaller models (referred to as student models). In thecontext of LLMs, KD involves using the original LLMs asteacher models to distill smaller LMs. Numerous studieshave focused on effectively transferring various abilities ofLLMs to smaller models. In this domain, methods can becategorized into two main types: white-box KD and black-box KD (as illustrated in Fig. 12).

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-11/924f71b2-d0c7-4bf4-8165-7633ffb6c93d/1527e4d0502c42068a76488d55c12d5803ac3ca395fde56562b1bf3ae9d3de54.jpg)



Fig. 12. Illustration of White-Box KD (left) and Black-Box KD (right).


White-box KD. White-box KD refers to distillation methodsthat leverage access to the structure and parameters of theteacher models. This approach enables KD to effectivelyutilize the intermediate features and output logits of theteacher models for enhanced performance of the studentmodels. MiniLLM [131] proposes to adopt the standardwhite-box KD approach but replace the forward Kullback-Leibler divergence (KLD) with the reverse KLD. GKD [132]introduces the use of on-policy data, which includes outputsequences generated by the student model itself, to furtherdistill the student model. This method focuses on aligningthe output logits between the teacher and student modelsusing these on-policy data. TED [133] presents a task-awarelayer-wise KD method. This approach involves adding fil-ters after each layer in both the teacher and student models,training these task-specific filters, and subsequently freez-ing the teacher model’s filters while training the studentfilters to align their output features with the correspond-ing teacher filters. MiniMoE [135] mitigates the capacitygap by utilizing a Mixture-of-Experts (MoE) model as thestudent model. DynaBERT [136] proposes to progressivelydecrease the models’ width and depth, and uses knowledgedistillation to train the smaller models. For newly emergingentities, pre-trained language models (LLMs) may lack up-to-date information. To address this, one solution involvesincorporating additional retrieved texts into prompts, albeitat an increased inference cost. Alternatively, KPTD [137]suggests transferring knowledge from entity definitions intoLLM parameters via knowledge distillation. This methodgenerates a transfer set based on entity definitions anddistills the student model to match output distributions withthe teacher model based on these definitions.

Black-box KD. Black-box KD refers to the knowledge dis-tillation methods in which the structure and parameters ofteacher models are not available. Typically, black-box KDonly uses the final results obtained by the teacher modelsto distill the student models. In the field of LLMs, black-box KD mainly guides the student models to learn LLMs’generalization ability and emergent ability, including In-Context Learning (ICL) ability [45], Chain-of-Thought (CoT)reasoning ability [14] and Instruction Following (IF) abil-ity [230].

Regarding the ICL ability, Multitask-ICT [118] introducesin-context learning distillation to transfer the multitask few-

shot ability of Large Language Models (LLMs), leveragingboth in-context learning and language modeling proficiency.MCKD [119] observes that student models distilled from in-context learned teacher models often exhibit superior per-formance on unseen input prompts. Building on this obser-vation, MCKD devises a multi-stage distillation paradigmwhere the student model from previous stages is employedto generate distillation data for subsequent stages, enhanc-ing the effectiveness of the distillation method.

To distill the Chain-of-Thought (CoT) reasoning ability,several techniques such as Distilling Step-by-Step [120],SCoTD [121], CoT Prompting [122], MCC-KD [123], andFine-tune-CoT [124] propose distillation methods that in-corporate responses and rationales extracted from LLMsto train student models. Socratic CoT [125] also targetsreasoning ability transfer to smaller models. Specifically,it fine-tunes a pair of student models, namely a QuestionGeneration (QG) model and a Question Answering (QA)model. The QG model is trained to generate intermediatequestions based on input questions, guiding the QA modelin producing the final response. PaD [126] observes thatfaulty reasoning (i.e., correct final answer but incorrectreasoning steps) can be detrimental to student models. Toaddress this, PaD proposes generating synthetic programsfor reasoning problems, which can then be automaticallychecked by an additional interpreter. This approach helps inremoving distillation data with faulty reasoning, enhancingthe quality of the training data for student models.

For the IF ability, several methods have been proposedto transfer this capability to smaller models. DISCO [128]introduces a technique where phrasal perturbations are gen-erated using a LLM. These perturbations are then filtered bya task-specific teacher model to distill high-quality counter-factual data. LaMini-LM [129] aims to transfer instructionfollowing ability by designing a diverse instruction set fordistilling student models. Lion [130] utilizes the teachermodel to identify difficult instructions, and generates newand complex instructions to distill the small model.

# 5.2.5 Dynamic Inference

Dynamic inference involves the adaptive selection of modelsub-structures during the inference process, conditionedon input data. This section focuses on early exiting tech-niques, which enable a LLM to halt its inference at differentmodel layers depending on specific samples or tokens.Notably, while MoE techniques (discussed in Sec. 5.1.1)also adjust model structure during inference, they typicallyinvolve expensive pre-training cost. In contrast, early ex-iting techniques only require training a small module todetermine when to conclude the inference. Some previoussurveys [231], [232] have reviewed dynamic inference tech-niques for traditional language models (e.g., RNN, LSTM).In this paper, we categorize studies on early exiting tech-niques for LLMs into two main types: sample-level earlyexiting and token-level early exiting (illustrated in Fig. 13).

Sample-level. Sample-level early exiting techniques focuson determining the optimal size and structure of LanguageModels (LLMs) for individual input samples. A commonapproach is to augment LLMs with additional modules aftereach layer, leveraging these modules to decide whether to

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-11/924f71b2-d0c7-4bf4-8165-7633ffb6c93d/cb0a87881d7b0ca5af00f9b4c655d88fee30855b934b4b06b4f21e3f0bf11a8b.jpg)



Fig. 13. Illustration of Token-level (up) and Sample-level (down) dynamicinference.


terminate inference at a specific layer. FastBERT [112], Dee-BERT [115], MP [233], and MPEE [113] train these modulesdirectly to make decisions (e.g., outputting 0 to continue or1 to stop) based on features from the current layer. GlobalPast-Future Early Exit [114] proposes a method that enrichesthe input to these modules with linguistic information fromboth preceding and subsequent layers. Given that futurelayer features are not directly accessible during inference, asimple feed-forward layer is trained to estimate these futurefeatures. PABEE [116] trains the modules as output headsfor direct prediction, suggesting inference termination whenpredictions remain consistent. HASHEE [117] employs anon-parametric decision-making approach based on the hy-pothesis that similar samples should exit inference at thesame layer.

Token-level. In the decoding stage of LLM inference, wheretokens are generated sequentially, token-level early exitingtechniques aim to optimize the size and structure of LLMsfor each output token. CALM [110] introduces early exitclassifiers after each Transformer layer, training them tooutput confidence scores that determine whether to haltinference at a specific layer. Notably, in the self-attentionblock, computing the current token’s feature at each layerrelies on all previous tokens’ features (i.e., KV cache) inthe same layer. To address the issue of missing KV cachedue to early exiting of previous tokens, CALM proposesdirectly copying the feature from the exiting layer to subse-quent layers, with experimental results showing only minorperformance degradation. SkipDecode [111] addresses lim-itations of previous early exiting methods that hinder theirapplicability to batch inference and KV caching, thereby lim-iting actual speed-up gains. For batch inference, SkipDecodeproposes a unified exit point for all tokens within a batch.Regarding KV caching, SkipDecode ensures a monotonicdecrease in exit points to prevent recomputation of KVcache, facilitating efficiency gains during inference.

# 5.3 Knowledge, Suggestions and Future Direction

In the field of efficient structure design, the pursuit ofalternative architectures to Transformers is a burgeoningarea of research. Examples such as Mamba [75], RWKV [62],and their respective variants [103], [106] have demonstratedcompetitive performance across various tasks, garnering

increasing attention in recent times. Nevertheless, it remainspertinent to investigate whether these non-Transformermodels may exhibit certain shortcomings compared toTransformer models. Concurrently, exploring the integra-tion of non-Transformer architectures with the attentionoperation [76], [105], [234] represents another promisingavenue for future research.

In the realm of model compression, quantization standsout as the predominant method employed in Large Lan-guage Model (LLM) deployment, primarily due to two keyfactors. Firstly, quantization presents a convenient means ofcompressing LLMs. For instance, employing Post-TrainingQuantization (PTQ) methods can reduce the parametercount of an LLM with seven billion parameters to a com-pressed form within a matter of minutes. Secondly, quan-tization holds the potential to achieve substantial reduc-tions in memory consumption and inference speed, whileintroducing only minor performance trade-offs. This com-promise is generally deemed acceptable for numerous real-world applications. However, it’s worth noting that quan-tization may still compromise certain emergent abilitiesof LLMs, such as self-calibration or multi-step reasoning.Additionally, in specific scenarios like dealing with longcontexts, quantization could lead to significant performancedegradation [214]. Consequently, it is required to carefullyselect appropriate quantization methods to mitigate the riskof such degradation in these specialized cases.

Extensive literature has devoted into studying sparse at-tention techniques for efficient long-context processing. Forexample, a recent representative work, StreamingLLM [151],can process 4 million tokens by only restoring severalattention sink tokens. Nonetheless, these approaches of-ten sacrifice critical information, resulting in performancedegradation. Therefore, the challenge of preserving essen-tial information while efficiently managing long contextsremains an important area for future exploration. As forthe weight pruning techniques, LLM-KICK [235] notes thatcurrent state-of-the-art (SOTA) methods experience con-siderable performance degradation even at relatively lowsparsity ratios. Consequently, developing effective weightpruning methods to maintain LLM performance remains anemerging and critical research direction.

The optimization of model structures often involves theuse of Neural Architecture Search (NAS), which typicallydemands extensive computational resources, posing a po-tential barrier to its practical application in compressingLLMs. Therefore, investigating the feasibility of employ-ing automatic structure optimization for LLM compressionwarrants further exploration. Additionally, the challengeremains for techniques like low-rank factorization (LRF) toachieve an optimal balance between compression ratio andtask performance. For instance, ASVD [149] achieves only amodest $1 0 \%$ to $2 0 \%$ compression ratio without compromis-ing the reasoning capabilities of LLMs.

In addition to employing individual model compres-sion techniques, several studies explore the combinationof different methods to compress LLMs, leveraging theirrespective advantages for improved efficiency. For instance,MPOE [90] applies weight matrix factorization specificallyto the expert Feed-Forward Networks (FFNs) in MoE-basedLLMs, with the goal of further reducing memory require-

ments. LLM-MQ [201] utilizes weight sparsity techniques toprotect weight outliers during model quantization, therebyminimizing quantization errors. LPLR [146] focuses onquantizing low-rank factorized weight matrices to furtherdecrease memory footprint and memory access cost duringLLM inference. Furthermore, LoSparse [145] combines low-rank factorization with weight pruning, leveraging pruningto enhance the diversity of low-rank approximation whileusing low-rank factorization to retain important weightsand prevent loss of critical information. These approacheshighlight the potential of integrating multiple compressiontechniques to achieve better optimization of LLMs.

# 6 SYSTEM-LEVEL OPTIMIZATION

The system-level optimization for LLM inference primarilyinvolves enhancing the model forward pass. Consideringthe computational graph of an LLM, there exist multipleoperators, with attention and linear operators dominatingmost of the runtime. As mentioned in Sec. 2.3, system-leveloptimization primarily considers the distinctive characteris-tics of the attention operator and the decoding approachwithin LLM. In particular, to address the specific issuesrelated to the decoding approach of LLMs, the linear opera-tor requires special tiling designs, and speculative decodingmethods are proposed to improve the utilization. The sub-stantial memory demand of LLMs leads to the offloadingof parameters or KV cache to the CPU. Furthermore, inthe context of online serving, requests come from multipleusers. Therefore, beyond the optimizations discussed earlier,online serving faces challenges related to memory, batching,and scheduling arising from asynchronous requests.

# 6.1 Inference Engine

The optimizations for inference engines are dedicated toaccelerating the model forward process. The main operatorsand the computational graph in LLM inference are highlyoptimized. Besides, the speculative decoding technique isproposed to accelerate the inference speed without perfor-mance degradation, and the offloading technique is intro-duced to mitigate the memory pressure.

# 6.1.1 Graph and Operator Optimization

Runtime Profiling. Using HuggingFace [260] implementa-tion, we profile the inference runtime with different mod-els and context lengths. The profiling results in Fig. 15demonstrate that attention operators and linear operatorscollectively dominate runtime, with their combined dura-tion often exceeding $7 5 \%$ of the inference duration. Conse-quently, a significant portion of optimization efforts at theoperator level is dedicated to enhancing the performance ofthe two operators. Furthermore, there are multiple operatorsoccupying a small proportion of runtime, which fragmentsthe operator execution timeline and increases the cost ofkernel launch on the CPU side. To address this issue, atthe computational graph level, current optimized inferenceengines implement highly fused operators.

Attention Operator Optimization. The standard attentioncomputation (e.g., using Pytorch) involves the multiplica-tion of the Query matrix (Q) with the Key matrix (K),

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-11/924f71b2-d0c7-4bf4-8165-7633ffb6c93d/3013587f2f789752986550a8cbc7b1a5449ed12899c64783d6edd0709f6c727e.jpg)



Fig. 14. Taxonomy of the optimization for LLM inference engine.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-11/924f71b2-d0c7-4bf4-8165-7633ffb6c93d/a870f56b05e966c3d0166419da7c76e229602b4cb4d2ae6a15e5f2dc1b671428.jpg)



(a) Llama2-7B,128 context length


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-11/924f71b2-d0c7-4bf4-8165-7633ffb6c93d/b1f8604d601927759ac5ca2e3b1daee2a985f5028f78919ab88fa0fd8f9cf888.jpg)



(b) Llama2-7B,2k context length


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-11/924f71b2-d0c7-4bf4-8165-7633ffb6c93d/722807b1b4e18ddfeccddd51d299e6c0d65cbb445f7917968112e12a0fe62f27.jpg)



(c) Baichuan2-13B,128 context length


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-11/924f71b2-d0c7-4bf4-8165-7633ffb6c93d/8e53f9f64ae450792b02fa0e18af1b89cc7990354dd11dece9e0cc136afe7846.jpg)



(d) Baichuan2-13B,2k context length


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-11/924f71b2-d0c7-4bf4-8165-7633ffb6c93d/097ff1852efb313eb9d9f017a09b5d264a1d7edd7fba84cc55cd7fd5256b86a0.jpg)



(e) Mixtral-8x7B,128 context length


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-11/924f71b2-d0c7-4bf4-8165-7633ffb6c93d/ae769b8d23b04271ed4da35d28d8c774ce75970acc40cfe9e12c3676a5cb6907.jpg)



(f) Mixtral-8x7B,2k context length



Fig. 15. Inference runtime breakdown over multiple LLMs.


resulting in quadratic time and space complexity in relationto the input sequence length. As shown in Fig. 15, thetime proportion of the attention operator increases as thecontext length grows. This translates to high demands onmemory size and computational capability, especially whendealing with long sequences. To address the computationaland memory overhead of standard attention computationon GPUs, customized attention operators are essential.FlashAttention [255], [256] fuses the entire attention oper-ation into a single, memory-efficient operator to alleviatememory access overhead. The input matrices (Q, K, V) andattention matrix are tiled into multiple blocks, which elimi-nates the need for complete data loading. Built upon FlashAttention, FlashDecoding [259] aims to maximize compu-tational parallelism for decoding. Due to the application ofthe decoding approach, the Q matrix degrades into a batchof vectors during decoding, which makes it challengingto fill the computational units if the parallelism is limitedto the batch size dimension. FlashDecoding addresses thisby introducing parallel computation along the sequencedimension. While this introduces some synchronizationoverhead to softmax computation, it leads to noticeableimprovements in parallelism, particularly for small batchsizes and long sequences. The subsequent work, FlashDe-coding $^ { + + }$ [253], observes that in previous works [255], [256],[259], the maximum value within the softmax only serves asa scaling factor to prevent data overflow. However, the dy-namical maximum value incurs significant synchronizationoverhead. Moreover, extensive experiments indicate thatin typical LLM (e.g., Llama2 [261], ChatGLM [262]), over

$9 9 . 9 9 \%$ of the softmax inputs fall within a certain range.Thus, FlashDecoding $^ { + + }$ proposes to determine the scalingfactor based on statistics in advance. This eliminates thesynchronization overhead in softmax computation, enablingparallel execution of subsequent operations alongside thesoftmax computation.

Linear Operator Optimization The linear operator playsa pivotal role in LLM inference, performing in featureprojection and Feedforward Neural Networks (FFNs). Intraditional neural networks, linear operators can be ab-stracted into General Matrix-Matrix Multiplication (GEMM)operations. However, in the case of LLM, the application ofthe decoding approach results in a notably reduced dimen-sion, diverging from the conventional GEMM workload.The low-level implementation of traditional GEMM hasbeen highly optimized, and mainstream LLM frameworks(e.g., DeepSpeed [258], vLLM [51], OpenPPL [263] and etc.)primarily call the GEMM APIs offered by cuBLAS [264]for linear operators. Without an explicitly tailored imple-mentation for GEMMs with a reduced dimension, the lin-ear operators during decoding suffer inefficiency. A no-table trend to address the issue is observed in the latestrelease of TensorRT-LLM [222]. It introduces a dedicatedGeneral Matrix-Vector Multiplication (GEMV) implemen-tation, potentially improving efficiency for the decodingstep. Recent research FlashDecoding $^ { + + }$ [253] makes a fur-ther step, addressing the inefficiency of cuBLAS [264] andCUTLASS [265] libraries when dealing with small batchsizes during the decode step. The authors first introducethe concept of the FlatGEMM operation to represent the

workload of GEMM with a highly reduced dimension (di-mension size $< 8$ in FlashDecoding $^ { + + }$ ). As FlatGEMM posesnew computational characteristics, the tiling strategy fortraditional GEMMs necessitates modification to be applied.The authors observe that two challenges exist as the work-load varies: low parallelism and memory access bottleneck.To tackle the challenges, FlashDecoding $^ { + + }$ adopts a fine-grained tiling strategy to improve parallelism, and leveragesthe double buffering technique to hide memory access la-tency. Furthermore, recognizing that the linear operations intypical LLM (e.g., Llama2 [261], ChatGLM [262]) often havefixed shapes, FlashDecoding $^ { + + }$ establishes a heuristic selec-tion mechanism. This mechanism dynamically chooses be-tween different linear operators based on the input size. Theoptions include FastGEMV [266], FlatGEMM, and GEMMprovided by cuBLAS [264], [265] libraries. This approachensures the selection of the most efficient operator for thegiven linear workload, potentially leading to better end-to-end performance.

Recently, the application of the MoE FFN to enhance themodel capability has become a trend in LLMs [12]. Thismodel structure also puts forward new requirements foroperator optimization. As shown in Fig. 15, in the Mixtralmodel with MoE FFN, the linear operator dominates theruntime due to the non-optimized FFN computation in theHuggingFace implementation. Besides, Mixtral’s adoptionof the GQA attention structure decreases the attention op-erator’s runtime proportion, which further points out theurgent need to optimize the FFN layer. MegaBlocks [254]is the first to optimize the computation for MoE FFN lay-ers. The work formulates the MoE FFN computation intoblock-sparse operations and proposes tailored GPU kernelsfor acceleration. However, MegaBlocks concentrates on theefficient training of the MoE models and hence ignores thecharacteristics of inference (e.g., the decoding approach).Existing frameworks are working hard to optimize thecomputations of the MoE FFN inference stage. The officialrepository of vLLM [51] integrates the fused kernels forMoE FFN in Triton [267], seamlessly removing the indexoverhead.

Graph-Level Optimization. Kernel fusion stands out as aprevalent graph-level optimization because of its capabil-ity to reduce runtime. There are three main advantagesof applying kernel fusion: (1) To reduce memory access.The fused kernel inherently removes the memory access ofintermediate results, mitigating the memory bottleneck foroperators. (2) To mitigate kernel launching overhead. Forsome lightweight operators (e.g., residual adding), the ker-nel launching time occupies most of the latency, and kernelfusion reduces individual kernel launchings. (3) To enhanceparallelism. For those operators without data dependency,when one-by-one kernel execution fails to fill the hardwarecapacity, it is beneficial to parallel the kernels via fusion.

The technique of kernel fusion proves effective withLLM inference, with all of the aforementioned benefits.FlashAttention [255] formulates the attention operator intoone single kernel, removing the overhead of accessing the at-tention results. Based on the fact that the attention operatoris memory-bounded, the reduction of memory access effec-tively transfers to runtime speed-up. ByteTransformer [257]and DeepSpeed [258] propose to fuse lightweight operators

including residual adding, layernorm, and activation func-tions, into the former linear operators to reduce the kernellaunching overhead. As a result, those lightweight operatorsdisappear in the timeline with nearly no extra latency. More-over, kernel fusion is also adopted to enhance the utilizationof LLM inference. The projections of Query, Key, and Valuematrices are originally three individual linear operations,and are fused into one linear operator to deploy on mod-ern GPUs. Currently, the kernel fusion technique has beenexploited in LLM inference practice, and highly optimizedinference engines employ only a few fused kernels withinthe runtime. For example, in FlashDecoding $^ { + + }$ [253] im-plementation, a transformer block integrates merely sevenfused kernels. Leveraging the aforementioned operators andkernel fusion optimization, FlashDecoding $^ { + + }$ achieves up to$4 . 8 6 \times$ speed-up over the HuggingFace implementation.

# 6.1.2 Speculative Decoding

Speculative decoding [268], [269] is an innovative decodingtechnique for auto-regressive LLMs designed to enhancedecoding efficiency without compromising the fidelity ofoutputs. The core idea of this approach involves employinga smaller model, termed a draft model, to predict severalsubsequent tokens efficiently, followed by validation ofthese predictions using the target LLM in parallel. Thismethodology aims to enable the LLM to generate multipletokens within the time frame typically required for a sin-gle inference. Fig. 16 demonstrates the comparison of thetraditional auto-regressive decoding method and the spec-ulative decoding approach. Formally, speculative decodingapproach consists of two steps:

1) Draft Construction: It employs the draft model to gen-erate several subsequent tokens, namely draft tokens,in parallel or in the auto-regressive manner.

2) Draft Verification: It employs the target model to com-pute the conditional probabilities of all the draft tokensin a single LLM inference step, subsequently determin-ing the acceptance of each draft token sequentially. Theacceptance rate, representing the average number ofaccepted draft tokens per inference step, serves as a keymetric for evaluating the performance of a speculativedecoding algorithm.

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-11/924f71b2-d0c7-4bf4-8165-7633ffb6c93d/28ec01d6f05cdcd94a84a21340f44f9754ef70e3c58283358973e5ea212aae7a.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-11/924f71b2-d0c7-4bf4-8165-7633ffb6c93d/5466594c2377cadcf5e8d28328e4bf1e2cb049f7fde1a3c6c33ac6ba7bf76721.jpg)



Fig. 16. Comparison of auto-regressive decoding (a) and speculativedecoding (b).


Speculative decoding ensures output equivalence withstandard auto-regressive decoding methods. Traditional de-coding techniques typically employ two primary sampling


TABLE 5Comparison of several open-source implementations of speculative decoding. In this table, we also show the additional overhead of constructingdraft models. Note that for SpD [236], [237], LADE [246], Medusa [50] and Eagle [247], we report the training cost from their original papers. Andfor SSD [239] and REST [29], we run the sub-LLM search and datastore construction with the code they provide, and report the time cost.Besides, for Medusa, we use Medusa-1 [50] which does not fine-tune the original LLM backbone.


<table><tr><td>Method</td><td>Draft Model</td><td>Draft Construction</td><td>Draft Verifier</td><td>Additional Overhead (GPU hours)</td><td>Acceptance Rate</td><td>Speed-up</td></tr><tr><td>SpD [236], [237]</td><td>small speculative model</td><td>one draft sequence</td><td>speculative sampling</td><td>275</td><td>1.77~2.02×</td><td>1.05~1.77×</td></tr><tr><td>LADE [246]</td><td>LLM + NGrams</td><td>one draft sequence</td><td>greedy sampling</td><td>0</td><td>1.92~2.14×</td><td>1.12~1.30×</td></tr><tr><td>SSD [239]</td><td>sub-LLM</td><td>one draft sequence</td><td>speculative sampling</td><td>4</td><td>1.64~1.74×</td><td>1.01~1.23×</td></tr><tr><td>REST [29]</td><td>datalstore</td><td>token tree</td><td>speculative sampling</td><td>1.5</td><td>2.18~2.31×</td><td>1.72~2.27×</td></tr><tr><td>Medusa-1 [50]</td><td>four LLM heads</td><td>token tree</td><td>speculative sampling</td><td>~24</td><td>2.52~2.62×</td><td>2.04~2.86×</td></tr><tr><td>Eagle [247]</td><td>one Transformer Layer</td><td>token tree</td><td>speculative sampling</td><td>96~192</td><td>3.47~3.72×</td><td>2.77~3.74×</td></tr></table>

strategies: greedy sampling and nucleus sampling. Greedysampling involves selecting the token with the highestprobability at each decoding step to generate a specificoutput sequence. The initial attempt at speculative decod-ing, known as Blockwise Parallel Decoding [270], aims toensure that the draft tokens precisely match the tokenssampled via greedy sampling, thus preserving output to-ken equivalence. In contrast, nucleus sampling involvessampling tokens from a probability distribution, resultingin diverse token sequences with each run. This diversitymakes nucleus sampling popular. To accommodate nucleussampling within speculative decoding frameworks, specu-lative sampling techniques [236], [237] have been proposed.Speculative sampling maintains output distribution equiv-alence, aligning with the probabilistic nature of nucleussampling to generate varied token sequences. Formally,given a sequence of tokens $x _ { 1 } , x _ { 2 } , . . . , x _ { n }$ and a sequence ofdraft tokens ${ \hat { x } } _ { n + 1 } , { \hat { x } } _ { n + 2 } , . . . , { \hat { x } } _ { n + k } ,$ the speculative samplingstrategy accepts the $i$ -th draft token with the followingprobabilities:

$$
\min  \left(1, \frac {p \left(\hat {x} _ {i} \mid x _ {1} , x _ {2} , \dots , x _ {i - 1}\right)}{q \left(\hat {x} _ {i} \mid x _ {1} , x _ {2} , \dots , x _ {i - 1}\right)}\right), \tag {12}
$$

where $p ( \cdot | \cdot )$ and $q ( \cdot | \cdot )$ denote the conditional probabilitiesfrom the target LLM and the draft model, respectively. Ifthe $i$ -th draft token is accepted, it sets $x _ { i } \gets \hat { x } _ { i }$ . Otherwise,it quits the verification of the following draft tokens, andresamples $x _ { i }$ from the following distribution:

$$
\operatorname {n o r m} \left(\max  \left(0, p \left(\cdot \mid x _ {1}, x _ {2}, \dots , x _ {i - 1}\right) - q \left(\cdot \mid x _ {1}, x _ {2}, \dots , x _ {i - 1}\right)\right)\right). \tag {13}
$$

Building upon speculative sampling, several variants [243],[248] have emerged, aimed at validating multiple drafttoken sequences. Notably, the token tree verifier [243] hasbecome a widely adopted verification strategy within thiscontext. This approach utilizes a tree-structured represen-tation of draft token sets and employs a tree attentionmechanism to efficiently perform the verification process.

In the speculative decoding approach, the acceptancerate of draft tokens is significantly influenced by the degreeto which the output distributions of draft models alignwith those of original LLMs. As a result, considerable re-search efforts have been directed towards improving thedesign of draft models. DistillSpec [238] directly distills asmaller draft model from the target LLM. SSD [239] involvesautomatically identifying a sub-model (a subset of modellayers) from the target LLM to serve as the draft model,

eliminating the need for separate training of the draft model.OSD [240] dynamically adjusts the output distribution of thedraft model to match the user query distribution in onlineLLM services. It achieves this by monitoring rejected drafttokens from the LLM and using this data to refine the draftmodel through distillation. PaSS [241] proposes utilizing thetarget LLM itself as the draft model, incorporating trainabletokens (look-ahead tokens) into the input sequence to enablesimultaneous generation of subsequent tokens. REST [242]introduces a retrieval-based speculative decoding approach,employing a non-parametric retrieval data store as the draftmodel. SpecInfer [243] introduces a collective boost-tuningtechnique to align the output distribution of a group ofdraft models with that of the target LLM. Lookahead decod-ing [246] involves generating n-grams of the target LLM inparallel to aid in generating draft tokens. Medusa [50] fine-tunes several heads of the LLM specifically for generatingsubsequent draft tokens. Eagle [247] adopts a lightweighttransformer layer called an auto-regression head to gener-ate draft tokens in an auto-regressive manner, integratingrich contextual features from the target LLM into the draftmodel’s input. Kangaroo [249] uses a fixed shallow sub-network as the draft model, and trains a lightweight adapteron the top of the sub-network. In this way, it does not needto train a separate draft model.

Another line of studies focuses on designing more effec-tive draft construction strategies. Conventional approachesoften yield single draft token sequences, posing challengesfor passing verification. In response, Spectr [248] advocatesgenerating multiple draft token sequences and employs a$k$ -sequential draft selection technique to concurrently verify$k$ sequences. This method leverages speculative sampling,ensuring equivalence in output distributions. Similarly,SpecInfer [243] adopts a comparable approach. However,unlike Spectr, SpecInfer merges draft token sequences into a“token tree” and introduces a tree attention mechanism forvalidation. This strategy is called the ”token tree verifier”.Due to its efficacy, token tree verifier has been widely em-braced in numerous speculative decoding algorithms [50],[242], [244], [247]. In addition to these efforts, Stage Spec-ulative Decoding [244] and Cascade Speculative Drafting(CS Drafting) [245] propose accelerating draft constructionby integrating speculative decoding directly into the tokengeneration process.

Comparative Experiments and Analysis. We conductan experiment to evaluate the speed-up performance of

the speculative decoding methods. Specifically, we thor-oughly review the studies of this field, and select sixof them that have open-sourced their codes, i.e., Spec-ulative Decoding (SpD) [236], [237], Lookahead Decod-ing (LADE) [246], REST [242], Self-speculative Decoding(SSD) [239], Medusa [50] and Eagle [247]. As for the eval-uation dataset, we use Vicuna-80 [7] to evaluate the abovemethods, which contains 80 questions that classified into10 categories. We report the average results on these 80questions. As for target LLMs, we adopt five fashion open-source LLMs, i.e., Vicuna-7B-V1.3 [7], Vicuna-13B-V1.3 [7],Vicuna-33B-V1.3 [7], LLaMA-2-7B [5] and LLaMA-2-13B [5].We report the range of evaluation metrics across these 5LLMs. As for draft models, we adopt two well-traineddraft models, i.e., LLaMA-68M and LLaMA-160M [243] forSpD. For other speculative decoding methods, we followtheir proposed draft construction approach and use thecheckpoints they provide. As for the evaluation metrics, weadopt acceptance rate, which denotes the ratio of the numberof accepted tokens to the number of generation steps, andspeed-up, which denotes the ratio of the latency of originalauto-regressive decoding to the latency of speculative de-coding when fixing the total length of output.

Tab. 5 provides a comparison of various speculativedecoding methods, highlighting several key observations:(1) Eagle demonstrates exceptional performance, achievinga notable $3 . 4 7 { \sim } 3 . 7 2 \times$ end-to-end speed-up across multipleLLMs. To understand its success, a deeper analysis of Eaglereveals two key factors. Firstly, Eagle employs an auto-regressive approach for decoding draft tokens, leveraginginformation from previously generated tokens directly. Sec-ondly, Eagle integrates rich features from previous tokens ofboth original LLMs and draft models to enhance the accu-racy of the next draft token generation. (2) The token treeverifier proves to be an effective technique in enhancing theperformance of speculative decoding methods. (3) The end-to-end speed-up achieved by these methods is often lowerthan the acceptance rate. This difference arises due to thepractical consideration that the generation cost associatedwith draft models cannot be overlooked.

# 6.1.3 Offloading

Current research investigates the potential of offloading toaccommodate the substantial memory demand of LLMs (seeSec. 2.3) in resource-constrained environments. The essenceof offloading is to offload part of the storage from the GPU tothe CPU when it is free of use. Intuitively, the focus of suchkind of research lies in hiding the expensive data move-ment latency between the GPU and the CPU. FlexGen [203]enables the offloading of weights, activations, and the KVcache, and further formulates a graph traversal problem foroffloading to maximize the throughput. The data loading ofthe next batch and the data storing of the previous batch canbe overlapped with the computation of the current batch.Another work llama.cpp [250] also assigns computationaltasks to the CPU, mitigating the data transfer overhead atthe cost of computing with the low-powered CPU. Powerin-fer [251] exploits the sparsity in activations using ReLU [271]in LLMs, and divides the activations into subsets of cold andhot neurons representing the frequency of computation. Thecold neurons are offloaded to the CPU for both storage and

computation in Powerinfer. Leveraging adaptive predictorsand sparse operators, Powerinfer significantly improves thecomputational efficiency with offloading. FastDecode [252]proposes to offload the storage and the computation of theentire attention operator to the CPU. Since the attentionoperation is computed on the CPU, the data movementof KV cache is reduced to merely some activations. Thenumber of CPUs is selected to match the workload latencyon GPUs so that the bubbles in the heterogeneous pipelineare mitigated.

# 6.2 Serving System

The optimizations for serving systemworks are dedicatedto improving the efficiency in handling asynchronous re-quests. The memory management is optimized to hold morerequests, and efficient batching and scheduling strategiesare integrated to enhance the system throughput. Besides,optimizations specific to distributed systems are proposedto exploit distributed computational resources.

# 6.2.1 Memory Management

The storage of KV cache dominates the memory usage inLLM serving, especially when the context length is long(see Sec. 2.3). Since the generation length is uncertain, itis challenging to allocate the space for KV cache storagein advance. Earlier implementations [286] usually allocatestorage space in advance based on the preset maximumlength of each request. However, in instances where re-quest generation is terminated early, this approach incurssignificant wastage of storage resources. To address theissue, $S ^ { 3 }$ [284] proposes to predict an upper bound of thegeneration length for each request, in order to reduce thewaste of the pre-allocated space. However, the static wayof KV cache memory allocation still fails when no suchlarge contiguous space exists. To deal with the fragmentedstorage, vLLM [51] proposes to store the KV cache in apaged manner following the operating system. vLLM firstallocates a memory space as large as possible and dividesit equally into multiple physical blocks. When a requestcomes, vLLM dynamically maps the generated KV cache tothe pre-allocated physical blocks in a discontinuous fashion.In this way, vLLM significantly reduces storage fragmenta-tion and achieves a higher throughput in LLM serving. Onthe basis of vLLM, LightLLM [278] uses a more fine-grainedKV cache storage to cut down the waste happening withthe irregular boundary. Instead of a block, LightLLM treatsthe KV cache of a token as a unit, so that the generated KVcache always saturates the pre-allocated space.

Current optimized service systems commonly employthis paged approach to manage the KV cache storage,thereby mitigating the waste of redundant KV cache mem-ory. However, the paged storage leads to irregular memoryaccess in the attention operator. For the attention operatorusing the paged KV cache, this necessitates the consider-ation of the mapping relationship between the virtual ad-dress space of the KV cache and its corresponding physicaladdress space. To enhance the efficiency of the attentionoperator, the loading pattern of the KV cache must be tai-lored to facilitate contiguous memory access. For instance,in the case of the PagedAttention by vLLM [51], the storage

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-11/924f71b2-d0c7-4bf4-8165-7633ffb6c93d/412dd97470c44a7f6bfc60bf4ebf5e6d03223e9d583dd7273ef45b89a29f4043.jpg)



Fig. 17. Taxonomy of the optimization for LLM serving system.


of the head size dimension is structured as a 16-byte con-tiguous vector for K cache, while FlashInfer [285] orches-trates diverse data layouts for the KV cache, accompaniedby an appropriately designed memory access scheme. Theoptimization of the attention operator in conjunction withpaged KV cache storage remains a forefront challenge in theadvancement of serving systems.

# 6.2.2 Continuous Batching

The request lengths in a batch can be different, leadingto low utilization when shorter requests are finished andlonger requests are still running. Due to the asynchronousnature of requests in serving scenarios, there exists anopportunity that such periods of low utilization could bemitigated. The continuous batching technique is proposedto leverage the opportunity by batching new requests oncesome old requests are finished. ORCA [277] is the first toutilize the continuous batching technique in LLM serving.The computation of each request encompasses multipleiterations, with each iteration representing either a pre-filling step or a decoding step. The author suggests thatdifferent requests can be batched at the iteration level. Thework implements iteration-level batching in linear oper-ators, concatenating different requests together in the se-quence dimension. Hence, the spare storage and computa-tional resources corresponding to the completed requestsare promptly released. Following ORCA, vLLM [51] ex-tends the technique to the attention computation, enablingrequests with different KV cache lengths to be batched to-gether. Sarathi [282], DeepSpeed-FastGen [279] and Sarathi-Serve [283] further introduce a split-and-fuse method tobatch together prefilling requests and decoding requests.Specifically, this method first splits the long prefilling re-quest in the sequence dimension, and then batches it to-gether with multiple short decoding requests. The split-and-fuse method balances the workloads among different itera-tions, and significantly reduces the tail latency via removingthe stalls from new requests. LightLLM [278] also adopts thesplit-and-fuse method.

The split-and-fuse technology operates on the premisethat requests during the prefilling stage can be partitionedinto discrete chunks. Chunked-prefill methodology involvessegmenting prefilling requests along the sequence dimen-sion, thereby preventing the potential bottlenecks for otherrequests. This strategy capitalizes on the auto-regressive

characteristics inherent in LLMs, where attention compu-tation only relies on prior tokens. Consequently, the math-ematical equivalence of chunked-prefill technology is guar-anteed, positioning it as a leading approach for reducingrequest latency in LLM serving.

# 6.2.3 Scheduling Strategy

In LLM serving, the job length of each request exhibitsvariability, and hence the order of executing requests sig-nificantly impacts the throughput of the serving system.The head-of-line blocking [280] happens when long requestsare accorded priority. Specifically, memory usage growsrapidly in response to long requests, impeding subsequentrequests when the system exhausts its memory capacity. Thepioneering work ORCA [277] and open-source systems, in-cluding vLLM [51] and LightLLM [278], employ the simplefirst-come-first-serve (FCFS) principle to schedule requests.DeepSpeed-FastGen [279] gives priority to the decoding re-quests to enhance the performance. FastServe [280] proposesa preemptive scheduling strategy to optimize the head-of-line blocking problem, achieving low job completion time(JCT) in LLM serving. FastServe employs a multi-levelfeedback queue (MLFQ) to prioritize the requests with theshortest remaining time. Since the auto-regressive decod-ing approach poses unknown request lengths, FastServepredicts the length first and utilizes a skip-join fashion tofind the proper priority for each request. Unlike previouswork, VTC [281] discusses the fairness in LLM serving.VTC introduces a cost function based on token numbers tomeasure fairness among clients, and further proposes a fairscheduler to ensure fairness.

# 6.2.4 Distributed Systems

In order to achieve high throughput, LLM services arecommonly deployed on distributed platforms. Recent workshave additionally focused on optimizing the performanceof such inference services by exploiting distributed char-acteristics. Notably, observing that the computations ofprefilling and decoding have interference with each other,splitwise [272], TetriInfer [273] and DistServe [274] demon-strate the efficiency of disaggregating the prefilling and thedecoding steps of a request. In this way, the two distinctsteps are processed independently based on their char-acteristics. ExeGPT [287] also adopts such disaggregatedarchitecture, and proposes different strategies with con-trollable variables to maximize system throughput under


TABLE 6Comparison of multiple open-source inference engines and serving systems. ”-” denotes no serving support. Note that the scheduling method ofTensorRT-LLM is not open-sourced.


<table><tr><td rowspan="2">Model</td><td colspan="4">Inference Optimization</td><td rowspan="2">Inference (token/s)</td><td colspan="3">Serving Optimization</td><td rowspan="2">Serving (req/s)</td></tr><tr><td>Attention</td><td>Linear</td><td>Graph</td><td>Speculative Decoding</td><td>Memory</td><td>Batching</td><td>Scheduling</td></tr><tr><td>HuggingFace [260]</td><td></td><td></td><td></td><td>✓</td><td>38.963</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>DeepSpeed [258]</td><td>✓</td><td></td><td>✓</td><td></td><td>80.947</td><td>blocked</td><td>split-and-fuse</td><td>decode prioritized</td><td>6.78</td></tr><tr><td>vLLM [51]</td><td>✓</td><td></td><td></td><td>✓</td><td>90.052</td><td>paged</td><td>continuous batching</td><td>prefill prioritized</td><td>7.11</td></tr><tr><td>OpenPPL [263]</td><td>✓</td><td></td><td>✓</td><td></td><td>81.169</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>FlashDecoding++ [253]</td><td>✓</td><td>✓</td><td>✓</td><td></td><td>106.636</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>LightLLM [278]</td><td>✓</td><td></td><td></td><td></td><td>73.599</td><td>token-wise</td><td>split-and-fuse</td><td>prefill prioritized</td><td>10.29</td></tr><tr><td>TensorRT-LLM [222]</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>92.512</td><td>paged</td><td>continuous batching</td><td>-</td><td>5.87</td></tr></table>

certain latency constraints. Llumnix [288] reschedules therequests at runtime for different serving objectives includingload balancing, de-fragmentation, and prioritization. Spot-Serve [275] is designed to provide LLM service on cloudswith preemptible GPU instances. SpotServe efficiently han-dles challenges including dynamic parallel control and in-stance migration, and also utilizes the auto-regressive natureof LLMs to achieve token-level state recovery. Moreover,Infinite-LLM [276] parallels different parts of the sequencein the attention operator across the data center, to ad-dress the challenges when serving extremely long contexts.LoongServe [289] proposes the elastic sequence parallelismto manage the elastic resource demand at the iteration level,reducing the data movement of KV cache via elaboratelydesigned scheduling.

# 6.3 Hardware Accelerator Design

Previous research efforts [290], [291], [292] have focused onoptimizing Transformer architectures, particularly enhanc-ing the attention operator, often employing sparse methodsto facilitate FPGA deployment. The FACT [293] accelera-tor achieves superior energy efficiency compared to theNVIDIA V100 GPU through mixed-precision quantizationfor linear operators and algorithm-hardware co-design, yetthese approaches are not tailored for generative LLMs.

Recent work like ALLO [294] highlights FPGA advan-tages in managing the memory-intensive decoding stageand emphasizes the importance of model compression tech-niques for LLMs’ efficient FPGA deployment. Conversely,DFX [295] focuses on decoding stage optimizations but lacksmodel compression methods, limiting scalability to largermodels and longer inputs (up to 1.5B model and 256 tokens).ALLO builds on these insights, further offering a libraryof High-level Synthesis (HLS) kernels that are composableand reusable. ALLO’s implementation demonstrates supe-rior generation speed-up compared to DFX in the prefillingstage, achieving enhanced energy efficiency and speedupover the NVIDIA A100 GPU during decoding.

FlightLLM [296] also leverages these insights, introduc-ing a configurable sparse digital signal processor (DSP)chain for various sparsity patterns with high computa-tional efficiency. It proposes an always-on-chip decodescheme with mixed-precision support to enhance mem-ory bandwidth utilization. FlightLLM achieves $6 . 0 \times$ higherenergy efficiency and $1 . 8 \times$ better cost efficiency than theNVIDIA V100S GPU for Llama2-7B models, with $1 . 2 \times$

higher throughput than the NVIDIA A100 GPU duringdecoding.

# 6.4 Comparison of LLM Frameworks

We compare the performance of multiple LLM frame-works in Table 6. The inference throughput is measuredwith Llama2-7B (batch size $^ { = 1 }$ , input length $\scriptstyle 1 = 1 \ k$ , outputlength $\scriptstyle 1 = 1 2 8$ ). The serving performance is the maximumthroughput measured on the ShareGPT [297] dataset. Bothare derived on a single NVIDIA A100 80GB GPU. Amongthe mentioned frameworks, DeepSpeed [258], vLLM [51],LightLLM [278] and TensorRT-LLM [222] integrate the serv-ing function to serve asynchronous requests from multipleusers. We also list the optimizations for each framework inthe table. All the frameworks except HuggingFace imple-ment operator-level or graph-level optimizations to enhanceperformance, and some of them also support the speculativedecoding technique. Note that the speculative decodingtechnique is off when we measure the inference perfor-mance for all frameworks. The results of inference through-put show that FlashDecoding++ and TensorRT-LLM out-perform others with optimizations covering predominantoperators and the computational graph. From the aspect ofserving, all the frameworks use fine-grained and discontigu-ous storage for KV cache, and apply the continuous batchingtechniques to improve the system utilization. Unlike vLLMand LightLLM, DeepSpeed prioritizes the decoding requestsin scheduling, which means no new request is merged ifthere are enough existing decoding requests in the batch.

# 6.5 Knowledge, Suggestions and Future Direction

The system-level optimization improves efficiency whilebringing no accuracy degradation, hence becoming preva-lent in the LLM inference practice. The optimization forinference is also applicable to serving. Recently, the oper-ator optimization has been closely combined with practi-cal serving scenarios, e.g.,, RadixAttention [52] designedspecifically for prefix caching, and tree attention [243] toaccelerate speculative decoding verification. The iterating ofapplications and scenarios will continue to put forward newrequirements for operator development.

Given the multifaceted objectives inherent in real-worldserving systems, such as JCT, system throughput, and fair-ness, the design of scheduling strategies becomes corre-spondingly intricate. Within the domain of LLM serving,where the length of requests is indeterminate, extant litera-ture commonly relies on predictive mechanisms to facilitate

the design of scheduling strategies. However, the efficacyof current predictors [273] falls short of ideal standards,indicating the potential for refinement and optimization inserving scheduling strategy development.

# 7 DISCUSSIONS OF KEY APPLICATION SCENAR-IOS

Current research endeavors have made significant strides inexploring the boundaries of efficient LLM inference acrossvarious optimization levels. However, further studies arewarranted to enhance LLM efficiency in practical scenarios.We have provided promising future directions for opti-mization techniques at the data-level (Sec. 4.3), model-level(Sec. 5.3), and system-level (Sec. 6.5). In this section, wesummarize four critical scenarios: agent and multi-modelframework, long-context LLMs, edge scenario deployment,and security-efficiency synergy, and provide a broader dis-cussion on them.

Agent and Multi-Model Framework. As discussed inSec. 4.3, recent advancements in agent and multi-modelframeworks [55], [56], [57] have significantly improvedagents’ capabilities to handle complex tasks and humanrequests by harnessing the powerful abilities of LLMs. Theseframeworks, while increasing the computational demandsof LLMs, introduce more parallelism into the structure ofLLMs’ output content, thereby creating opportunities fordata-level and system-level optimizations such as output or-ganization techniques [52]. Furthermore, these frameworksnaturally introduce a new optimization level, i.e., pipeline-level, which holds potential for efficiency enhancements atthis level [58].

In addition, there is a growing research trend [298] fo-cused on extending AI agents into the multimodal domain,which often utilize Large Multimodal Models (LMMs) asthe core of these agent systems. To enhance the efficiency ofthese emerging LMM-based agents, designing optimizationtechniques for LMMs is a promising research direction.

Long-Context LLMs. Currently, LLMs face the challengeof handling increasingly longer input contexts. However,the self-attention operation, the fundamental componentof Transformer-style LLMs, exhibits quadratic complexityin relation to the context length, imposing constraints onmaximum context length during both training and infer-ence phases. Various strategies have been explored to ad-dress this limitation, including input compression (Sec. 4.1),sparse attention (Sec. 5.2.2), design of low-complexity struc-tures (Sec. 5.1.3), and optimization of attention opera-tors (Sec. 6.1.1). Notably, non-Transformer architectures(Sec. 5.1.3) with sub-quadratic or linear complexity haverecently garnered significant interest from researchers.

Despite their efficiency, the competitiveness of thesenovel architectures compared to the Transformer archi-tecture across various abilities, such as in-context learn-ing ability and long-range modeling ability, is still underscrutiny [76], [299]. Therefore, exploring the capabilities ofthese new architectures from multiple angles and address-ing their limitations remains a valuable pursuit. Moreover,it is crucial to determine the necessary context lengths forvarious scenarios and tasks, as well as identify the next-

generation architecture that will serve as the foundationalbackbone for LLMs in the future.

Edge Scenario Deployment. While considerable effortshave been directed towards enhancing the efficiency ofLLM inference, deploying LLMs onto extremely resource-constrained edge devices like mobile phones presents ongo-ing challenges. Recently, numerous researchers [300], [301],[302], [303], [304], [305], [306], [307], [308], [309], [310] haveshown interest in pre-training smaller language modelswith 1B to 3B parameters. Models of this scale offer re-duced resource costs during inference and hold potential forachieving generalization abilities and competitive perfor-mance compared to larger models. However, the methodsto develop such efficient and powerful smaller languagemodels remain under-explored.

Several studies have initiated this promising direction.For instance, MiniCPM [309] conducts sandbox experi-ments to determine optimal pre-training hyper-parameters.PanGu- $\pi$ -Pro [302] suggests initializing model weights frompre-trained LLMs using metrics and techniques from modelpruning. MobileLLM [310] adopts a“deep and thin” archi-tecture for small model design and proposes weight sharingacross different layers to increase the number of layerswithout additional memory costs. Nevertheless, a perfor-mance gap still exists between small and large models,necessitating future studies to narrow this gap. In the future,there is a crucial need for research aimed at identifying themodel scale limited in the edge scenarios, and exploring theboundaries of various optimization methods on designingsmaller models.

Beyond designing smaller models, system-level opti-mization offers a promising direction in LLM deployment. Anotable recent project, MLC-LLM [311], successfully deploysthe LLaMA-7B model on mobile phones. MLC-LLM pri-marily employs compilation techniques like fusion, memoryplanning, and loop optimization to enhance latency and re-duce memory cost during inference. Additionally, adoptingthe cloud-edge collaboration techniques, or designing moresophisticated hardware accelerators can also help deployLLMs onto edge devices.

Security-Efficiency Synergy. In addition to task perfor-mance and efficiency, security is also a crucial factor thatmust be considered in LLM applications [312], [313]. Cur-rent research primarily focuses on efficiency optimiza-tion without adequately addressing security considerations.Therefore, it is critical to investigate the interplay betweenefficiency and security and determine whether the currentoptimization techniques compromise the security of LLMs.If these techniques negatively impacts LLMs’ security, apromising direction would involve developing new opti-mization methods or refining the existing ones to achievea better trade-off between LLMs’ efficiency and security.

# 8 CONCLUSION

Efficient LLM inference focuses on reducing the compu-tational, memory access, and memory costs during LLMinference processes, aiming to optimize efficiency metricssuch as latency, throughput, storage, power, and energy.This survey offers a comprehensive review of efficient LLMinference research, presenting insights, recommendations,

and future directions for key techniques. Initially, we intro-duce a hierarchical taxonomy encompassing data-, model-, and system-level optimizations. Subsequently, guided bythis taxonomy, we meticulously examine and summarizestudies at each level and sub-field. For well-establishedtechniques like model quantization and efficient servingsystems, we conduct experiments to evaluate and analyzetheir performance. Based on these analyses, we offer practi-cal suggestions and identify promising research avenues forpractitioners and researchers in the field.

# ACKNOWLEDGEMENTS

This work was supported by National Natural ScienceFoundation of China (No. 62325405, 62104128, U19B2019,U21B2031, 61832007, 62204164), Tsinghua EE Xilinx AI Re-search Fund, and Beijing National Research Center for In-formation Science and Technology (BNRist). We thank forall the support from Infinigence-AI. We thank XiangshengShi, Zinan Lin, Xinhao Yang, Hongyi Wang, Linfeng Zhang,Yulin Wang, Xuemin Sun, Saiqian Zhang for their valuablesuggestions on the paper. We thank Shengxiang Wang, QiuliMao for providing the efficiency profiling data of quantizedoperators.

# REFERENCES



[1] A. Radford, K. Narasimhan, T. Salimans, I. Sutskever et al.,“Improving language understanding by generative pre-training,”2018.





[2] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskeveret al., “Language models are unsupervised multitask learners,”OpenAI blog, vol. 1, no. 8, p. 9, 2019.





[3] T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhari-wal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell et al.,“Language models are few-shot learners,” Advances in neuralinformation processing systems, vol. 33, pp. 1877–1901, 2020.





[4] S. Zhang, S. Roller, N. Goyal, M. Artetxe, M. Chen, S. Chen,C. Dewan, M. Diab, X. Li, X. V. Lin et al., “Opt: Open pre-trainedtransformer language models,” arXiv preprint arXiv:2205.01068,2022.





[5] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux,T. Lacroix, B. Roziere, N. Goyal, E. Hambro, F. Azhar ` et al.,“Llama: Open and efficient foundation language models,” arXivpreprint arXiv:2302.13971, 2023.





[6] A. Yang, B. Xiao, B. Wang, B. Zhang, C. Bian, C. Yin, C. Lv, D. Pan,D. Wang, D. Yan et al., “Baichuan 2: Open large-scale languagemodels,” arXiv preprint arXiv:2309.10305, 2023.





[7] W.-L. Chiang, Z. Li, Z. Lin, Y. Sheng, Z. Wu, H. Zhang, L. Zheng,S. Zhuang, Y. Zhuang, J. E. Gonzalez et al., “Vicuna: An open-source chatbot impressing gpt-4 with $9 0 \% ^ { * }$ chatgpt quality,” Seehttps://vicuna. lmsys. org (accessed 14 April 2023), 2023.





[8] D. Li, R. Shao, A. Xie, Y. Sheng, L. Zheng, J. Gonzalez, I. Stoica,X. Ma, and H. Zhang, “How long can context length of open-source llms truly promise?” in NeurIPS 2023 Workshop on Instruc-tion Tuning and Instruction Following, 2023.





[9] B. Workshop, T. L. Scao, A. Fan, C. Akiki, E. Pavlick, S. Ilic,´D. Hesslow, R. Castagne, A. S. Luccioni, F. Yvon ´ et al., “Bloom: A176b-parameter open-access multilingual language model,” arXivpreprint arXiv:2211.05100, 2022.





[10] E. Almazrouei, H. Alobeidli, A. Alshamsi, A. Cappelli, R. Cojo-caru, M. Debbah, E. Goffinet, D. Hesslow, J. Launay, Q. Malartic ´et al., “The falcon series of open language models,” arXiv preprintarXiv:2311.16867, 2023.





[11] Z. Du, Y. Qian, X. Liu, M. Ding, J. Qiu, Z. Yang, and J. Tang,“Glm: General language model pretraining with autoregressiveblank infilling,” arXiv preprint arXiv:2103.10360, 2021.





[12] A. Q. Jiang, A. Sablayrolles, A. Roux, A. Mensch, B. Savary,C. Bamford, D. S. Chaplot, D. d. l. Casas, E. B. Hanna, F. Bressandet al., “Mixtral of experts,” arXiv preprint arXiv:2401.04088, 2024.





[13] J. Yang, H. Jin, R. Tang, X. Han, Q. Feng, H. Jiang, S. Zhong,B. Yin, and X. Hu, “Harnessing the power of llms in practice: Asurvey on chatgpt and beyond,” ACM Transactions on KnowledgeDiscovery from Data, 2023.





[14] J. Wei, X. Wang, D. Schuurmans, M. Bosma, F. Xia, E. Chi, Q. V. Le,D. Zhou et al., “Chain-of-thought prompting elicits reasoning inlarge language models,” Advances in Neural Information ProcessingSystems, vol. 35, pp. 24 824–24 837, 2022.





[15] M. Chen, J. Tworek, H. Jun, Q. Yuan, H. P. d. O. Pinto, J. Kaplan,H. Edwards, Y. Burda, N. Joseph, G. Brockman et al., “Eval-uating large language models trained on code,” arXiv preprintarXiv:2107.03374, 2021.





[16] S. Bubeck, V. Chandrasekaran, R. Eldan, J. Gehrke, E. Horvitz,E. Kamar, P. Lee, Y. T. Lee, Y. Li, S. Lundberg et al., “Sparksof artificial general intelligence: Early experiments with gpt-4,”arXiv preprint arXiv:2303.12712, 2023.





[17] X. Zhu, J. Li, Y. Liu, C. Ma, and W. Wang, “A survey onmodel compression for large language models,” arXiv preprintarXiv:2308.07633, 2023.





[18] S. Park, J. Choi, S. Lee, and U. Kang, “A comprehensive surveyof compression algorithms for language models,” arXiv preprintarXiv:2401.15347, 2024.





[19] W. Wang, W. Chen, Y. Luo, Y. Long, Z. Lin, L. Zhang, B. Lin,D. Cai, and X. He, “Model compression and efficient infer-ence for large language models: A survey,” arXiv preprintarXiv:2402.09748, 2024.





[20] Y. Tang, Y. Wang, J. Guo, Z. Tu, K. Han, H. Hu, andD. Tao, “A survey on transformer compression,” arXiv preprintarXiv:2402.05964, 2024.





[21] T. Ding, T. Chen, H. Zhu, J. Jiang, Y. Zhong, J. Zhou, G. Wang,Z. Zhu, I. Zharkov, and L. Liang, “The efficiency spectrum oflarge language models: An algorithmic survey,” arXiv preprintarXiv:2312.00678, 2023.





[22] X. Miao, G. Oliaro, Z. Zhang, X. Cheng, H. Jin, T. Chen,and Z. Jia, “Towards efficient generative large language modelserving: A survey from algorithms to systems,” arXiv preprintarXiv:2312.15234, 2023.





[23] Z. Wan, X. Wang, C. Liu, S. Alam, Y. Zheng, Z. Qu, S. Yan, Y. Zhu,Q. Zhang, M. Chowdhury et al., “Efficient large language models:A survey,” arXiv preprint arXiv:2312.03863, vol. 1, 2023.





[24] M. Xu, W. Yin, D. Cai, R. Yi, D. Xu, Q. Wang, B. Wu, Y. Zhao,C. Yang, S. Wang et al., “A survey of resource-efficient llm andmultimodal foundation models,” arXiv preprint arXiv:2401.08092,2024.





[25] W. X. Zhao, K. Zhou, J. Li, T. Tang, X. Wang, Y. Hou, Y. Min,B. Zhang, J. Zhang, Z. Dong et al., “A survey of large languagemodels,” arXiv preprint arXiv:2303.18223, 2023.





[26] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N.Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,”Advances in neural information processing systems, vol. 30, 2017.





[27] Z. Yuan, Y. Shang, Y. Zhou, Z. Dong, C. Xue, B. Wu, Z. Li,Q. Gu, Y. J. Lee, Y. Yan et al., “Llm inference unveiled: Survey androofline model insights,” arXiv preprint arXiv:2402.16363, 2024.





[28] A. Golden, S. Hsia, F. Sun, B. Acun, B. Hosmer, Y. Lee, Z. DeVito,J. Johnson, G.-Y. Wei, D. Brooks et al., “Is flash attention stable?”arXiv preprint arXiv:2405.02803, 2024.





[29] P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal,H. Kuttler, M. Lewis, W.-t. Yih, T. Rockt ¨ aschel ¨ et al., “Retrieval-augmented generation for knowledge-intensive nlp tasks,” Ad-vances in Neural Information Processing Systems, vol. 33, pp. 9459–9474, 2020.





[30] A. Chevalier, A. Wettig, A. Ajith, and D. Chen, “Adapt-ing language models to compress contexts,” arXiv preprintarXiv:2305.14788, 2023.





[31] W. Shi, S. Min, M. Yasunaga, M. Seo, R. James, M. Lewis,L. Zettlemoyer, and W. tau Yih, “Replug: Retrieval-augmentedblack-box language models,” 2023.





[32] A. Asai, Z. Wu, Y. Wang, A. Sil, and H. Hajishirzi, “Self-rag: Learning to retrieve, generate, and critique through self-reflection,” 2023.





[33] D. Wingate, M. Shoeybi, and T. Sorensen, “Prompt compres-sion and contrastive conditioning for controllability and toxicityreduction in language models,” arXiv preprint arXiv:2210.03162,2022.





[34] J. Mu, X. L. Li, and N. Goodman, “Learning to compress promptswith gist tokens,” arXiv preprint arXiv:2304.08467, 2023.





[35] T. Ge, J. Hu, X. Wang, S.-Q. Chen, and F. Wei, “In-contextautoencoder for context compression in a large language model,”arXiv preprint arXiv:2307.06945, 2023.





[36] F. Xu, W. Shi, and E. Choi, “Recomp: Improving retrieval-augmented lms with compression and selective augmentation,”arXiv preprint arXiv:2310.04408, 2023.





[37] W. Fei, X. Niu, P. Zhou, L. Hou, B. Bai, L. Deng, and W. Han, “Ex-tending context window of large language models via semanticcompression,” arXiv preprint arXiv:2312.09571, 2023.





[38] W. Zhou, Y. E. Jiang, R. Cotterell, and M. Sachan, “Efficientprompting via dynamic in-context learning,” arXiv preprintarXiv:2305.11170, 2023.





[39] Y. Li, B. Dong, F. Guerin, and C. Lin, “Compressing contextto enhance inference efficiency of large language models,” inProceedings of the 2023 Conference on Empirical Methods in NaturalLanguage Processing, 2023, pp. 6342–6353.





[40] F. Yin, J. Vig, P. Laban, S. Joty, C. Xiong, and C.-S. J. Wu, “Did youread the instructions? rethinking the effectiveness of task defi-nitions in instruction learning,” arXiv preprint arXiv:2306.01150,2023.





[41] H. Jung and K.-J. Kim, “Discrete prompt compression withreinforcement learning,” arXiv preprint arXiv:2308.08758, 2023.





[42] H. Jiang, Q. Wu, C.-Y. Lin, Y. Yang, and L. Qiu, “Llmlingua:Compressing prompts for accelerated inference of large languagemodels,” in The 2023 Conference on Empirical Methods in NaturalLanguage Processing, 2023.





[43] H. Jiang, Q. Wu, X. Luo, D. Li, C.-Y. Lin, Y. Yang, andL. Qiu, “Longllmlingua: Accelerating and enhancing llms inlong context scenarios via prompt compression,” arXiv preprintarXiv:2310.06839, 2023.





[44] X. Huang, L. L. Zhang, K.-T. Cheng, and M. Yang, “Boosting llmreasoning: Push the limits of few-shot learning with reinforcedin-context pruning,” arXiv preprint arXiv:2312.08901, 2023.





[45] Q. Dong, L. Li, D. Dai, C. Zheng, Z. Wu, B. Chang, X. Sun, J. Xu,and Z. Sui, “A survey for in-context learning,” arXiv preprintarXiv:2301.00234, 2022.





[46] X. L. Li and P. Liang, “Prefix-tuning: Optimizing continuousprompts for generation,” in Proceedings of the 59th Annual Meetingof the Association for Computational Linguistics and the 11th Inter-national Joint Conference on Natural Language Processing (Volume 1:Long Papers), 2021, pp. 4582–4597.





[47] X. Ning, Z. Lin, Z. Zhou, H. Yang, and Y. Wang, “Skeleton-of-thought: Large language models can do parallel decoding,” arXivpreprint arXiv:2307.15337, 2023.





[48] S. Jin, Y. Wu, H. Zheng, Q. Zhang, M. Lentz, Z. M. Mao,A. Prakash, F. Qian, and D. Zhuo, “Adaptive skeleton graphdecoding,” arXiv preprint arXiv:2402.12280, 2024.





[49] M. Liu, A. Zeng, B. Wang, P. Zhang, J. Tang, and Y. Dong,“Apar: Llms can do auto-parallel auto-regressive decoding,”arXiv preprint arXiv:2401.06761, 2024.





[50] T. Cai, Y. Li, Z. Geng, H. Peng, J. D. Lee, D. Chen, and T. Dao,“Medusa: Simple llm inference acceleration framework with mul-tiple decoding heads,” 2024.





[51] W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu,J. Gonzalez, H. Zhang, and I. Stoica, “Efficient memory manage-ment for large language model serving with pagedattention,” inProceedings of the 29th Symposium on Operating Systems Principles,2023, pp. 611–626.





[52] L. Zheng, L. Yin, Z. Xie, J. Huang, C. Sun, C. H. Yu, S. Cao,C. Kozyrakis, I. Stoica, J. E. Gonzalez et al., “Efficiently pro-gramming large language models using sglang,” arXiv preprintarXiv:2312.07104, 2023.





[53] S. Yao, D. Yu, J. Zhao, I. Shafran, T. Griffiths, Y. Cao, andK. Narasimhan, “Tree of thoughts: Deliberate problem solvingwith large language models,” Advances in Neural InformationProcessing Systems, vol. 36, 2024.





[54] M. Besta, N. Blach, A. Kubicek, R. Gerstenberger, M. Podstawski,L. Gianinazzi, J. Gajda, T. Lehmann, H. Niewiadomski, P. Nyczyket al., “Graph of thoughts: Solving elaborate problems withlarge language models,” in Proceedings of the AAAI Conference onArtificial Intelligence, vol. 38, no. 16, 2024, pp. 17 682–17 690.





[55] Z. Xi, W. Chen, X. Guo, W. He, Y. Ding, B. Hong, M. Zhang,J. Wang, S. Jin, E. Zhou et al., “The rise and potential oflarge language model based agents: A survey,” arXiv preprintarXiv:2309.07864, 2023.





[56] Q. Sun, Z. Yin, X. Li, Z. Wu, X. Qiu, and L. Kong, “Corex:Pushing the boundaries of complex reasoning through multi-model collaboration,” arXiv preprint arXiv:2310.00280, 2023.





[57] T. Guo, X. Chen, Y. Wang, R. Chang, S. Pei, N. V. Chawla,O. Wiest, and X. Zhang, “Large language model based multi-agents: A survey of progress and challenges,” arXiv preprintarXiv:2402.01680, 2024.





[58] L. Chen, M. Zaharia, and J. Zou, “Frugalgpt: How to use largelanguage models while reducing cost and improving perfor-mance,” arXiv preprint arXiv:2305.05176, 2023.





[59] Y. Li, T. Cai, Y. Zhang, D. Chen, and D. Dey, “What makesconvolutional models great on long sequence modeling?” arXivpreprint arXiv:2210.09298, 2022.





[60] D. W. Romero, A. Kuzina, E. J. Bekkers, J. M. Tomczak, andM. Hoogendoorn, “Ckconv: Continuous kernel convolution forsequential data,” arXiv preprint arXiv:2102.02611, 2021.





[61] M. Poli, S. Massaroli, E. Nguyen, D. Y. Fu, T. Dao, S. Baccus,Y. Bengio, S. Ermon, and C. Re, “Hyena hierarchy: Towards larger ´convolutional language models,” in International Conference onMachine Learning. PMLR, 2023, pp. 28 043–28 078.





[62] B. Peng, E. Alcaide, Q. Anthony, A. Albalak, S. Arcadinho,H. Cao, X. Cheng, M. Chung, M. Grella, K. K. GV et al.,“Rwkv: Reinventing rnns for the transformer era,” arXiv preprintarXiv:2305.13048, 2023.





[63] Y. Sun, L. Dong, S. Huang, S. Ma, Y. Xia, J. Xue, J. Wang, andF. Wei, “Retentive network: A successor to transformer for largelanguage models,” arXiv preprint arXiv:2307.08621, 2023.





[64] A. Gu, T. Dao, S. Ermon, A. Rudra, and C. Re, “Hippo: Recurrent ´memory with optimal polynomial projections,” Advances in neuralinformation processing systems, vol. 33, pp. 1474–1487, 2020.





[65] A. Gu, I. Johnson, K. Goel, K. Saab, T. Dao, A. Rudra, andC. Re, “Combining recurrent, convolutional, and continuous- ´time models with linear state space layers,” Advances in neuralinformation processing systems, vol. 34, pp. 572–585, 2021.





[66] A. Gu, K. Goel, and C. Re, “Efficiently modeling long sequences ´with structured state spaces,” arXiv preprint arXiv:2111.00396,2021.





[67] A. Gupta, A. Gu, and J. Berant, “Diagonal state spaces are as ef-fective as structured state spaces,” Advances in Neural InformationProcessing Systems, vol. 35, pp. 22 982–22 994, 2022.





[68] A. Gu, K. Goel, A. Gupta, and C. Re, “On the parameterization ´and initialization of diagonal state space models,” Advances inNeural Information Processing Systems, vol. 35, pp. 35 971–35 983,2022.





[69] H. Mehta, A. Gupta, A. Cutkosky, and B. Neyshabur, “Longrange language modeling via gated state spaces,” in InternationalConference on Learning Representations, 2023.





[70] D. Y. Fu, T. Dao, K. K. Saab, A. W. Thomas, A. Rudra, and C. Re,´“Hungry hungry hippos: Towards language modeling with statespace models,” arXiv preprint arXiv:2212.14052, 2022.





[71] R. Hasani, M. Lechner, T.-H. Wang, M. Chahine, A. Amini, andD. Rus, “Liquid structural state-space models,” arXiv preprintarXiv:2209.12951, 2022.





[72] J. T. Smith, A. Warrington, and S. W. Linderman, “Simpli-fied state space layers for sequence modeling,” arXiv preprintarXiv:2208.04933, 2022.





[73] J. Pilault, M. Fathi, O. Firat, C. Pal, P.-L. Bacon, and R. Goroshin,“Block-state transformers,” Advances in Neural Information Pro-cessing Systems, vol. 36, 2024.





[74] J. Wang, J. N. Yan, A. Gu, and A. M. Rush, “Pretraining withoutattention,” arXiv preprint arXiv:2212.10544, 2022.





[75] A. Gu and T. Dao, “Mamba: Linear-time sequence modeling withselective state spaces,” arXiv preprint arXiv:2312.00752, 2023.





[76] J. Park, J. Park, Z. Xiong, N. Lee, J. Cho, S. Oymak, K. Lee,and D. Papailiopoulos, “Can mamba learn how to learn? acomparative study on in-context learning tasks,” arXiv preprintarXiv:2402.04248, 2024.





[77] N. Shazeer, “Fast transformer decoding: One write-head is allyou need,” arXiv preprint arXiv:1911.02150, 2019.





[78] J. Ainslie, J. Lee-Thorp, M. de Jong, Y. Zemlyanskiy, F. Lebron, ´and S. Sanghai, “Gqa: Training generalized multi-query trans-former models from multi-head checkpoints,” arXiv preprintarXiv:2305.13245, 2023.





[79] S. Wang, B. Z. Li, M. Khabsa, H. Fang, and H. Ma, “Lin-former: Self-attention with linear complexity,” arXiv preprintarXiv:2006.04768, 2020.





[80] G. I. Winata, S. Cahyawijaya, Z. Lin, Z. Liu, and P. Fung,“Lightweight and efficient end-to-end speech recognition usinglow-rank transformer,” in ICASSP 2020-2020 IEEE InternationalConference on Acoustics, Speech and Signal Processing (ICASSP).IEEE, 2020, pp. 6144–6148.





[81] A. Gupta, Y. Yuan, Y. Zhou, and C. Mendis, “Flurka: Fast fusedlow-rank & kernel attention,” arXiv preprint arXiv:2306.15799,2023.





[82] X. Ma, X. Kong, S. Wang, C. Zhou, J. May, H. Ma, and L. Zettle-moyer, “Luna: Linear unified nested attention,” Advances in Neu-ral Information Processing Systems, vol. 34, pp. 2441–2453, 2021.





[83] J. Lee, Y. Lee, J. Kim, A. Kosiorek, S. Choi, and Y. W. Teh,“Set transformer: A framework for attention-based permutation-invariant neural networks,” in International conference on machinelearning. PMLR, 2019, pp. 3744–3753.





[84] A. Katharopoulos, A. Vyas, N. Pappas, and F. Fleuret, “Trans-formers are rnns: Fast autoregressive transformers with linearattention,” in International conference on machine learning. PMLR,2020, pp. 5156–5165.





[85] K. M. Choromanski, V. Likhosherstov, D. Dohan, X. Song,A. Gane, T. Sarlos, P. Hawkins, J. Q. Davis, A. Mohiuddin,L. Kaiser et al., “Rethinking attention with performers,” in In-ternational Conference on Learning Representations, 2020.





[86] H. Peng, N. Pappas, D. Yogatama, R. Schwartz, N. Smith, andL. Kong, “Random feature attention,” in International Conferenceon Learning Representations, 2022.





[87] P. Kacham, V. Mirrokni, and P. Zhong, “Polysketchformer: Fasttransformers via sketches for polynomial kernels,” arXiv preprintarXiv:2310.01655, 2023.





[88] W. Fedus, B. Zoph, and N. Shazeer, “Switch transformers: Scalingto trillion parameter models with simple and efficient sparsity,”The Journal of Machine Learning Research, vol. 23, no. 1, pp. 5232–5270, 2022.





[89] Z. Zhang, Y. Lin, Z. Liu, P. Li, M. Sun, and J. Zhou, “Moefication:Transformer feed-forward layers are mixtures of experts,” inFindings of the Association for Computational Linguistics: ACL 2022,2022, pp. 877–890.





[90] Z.-F. Gao, P. Liu, W. X. Zhao, Z.-Y. Lu, and J.-R. Wen, “Parameter-efficient mixture-of-experts architecture for pre-trained languagemodels,” in Proceedings of the 29th International Conference onComputational Linguistics, 2022, pp. 3263–3273.





[91] A. Komatsuzaki, J. Puigcerver, J. Lee-Thorp, C. R. Ruiz,B. Mustafa, J. Ainslie, Y. Tay, M. Dehghani, and N. Houlsby,“Sparse upcycling: Training mixture-of-experts from densecheckpoints,” arXiv preprint arXiv:2212.05055, 2022.





[92] M. Lewis, S. Bhosale, T. Dettmers, N. Goyal, and L. Zettlemoyer,“Base layers: Simplifying training of large, sparse models,” inInternational Conference on Machine Learning. PMLR, 2021, pp.6265–6274.





[93] Y. Zhou, T. Lei, H. Liu, N. Du, Y. Huang, V. Zhao, A. M.Dai, Q. V. Le, J. Laudon et al., “Mixture-of-experts with expertchoice routing,” Advances in Neural Information Processing Systems,vol. 35, pp. 7103–7114, 2022.





[94] B. Zoph, I. Bello, S. Kumar, N. Du, Y. Huang, J. Dean, N. Shazeer,and W. Fedus, “St-moe: Designing stable and transferable sparseexpert models,” arXiv preprint arXiv:2202.08906, 2022.





[95] D. Dai, L. Dong, S. Ma, B. Zheng, Z. Sui, B. Chang, and F. Wei,“Stablemoe: Stable routing strategy for mixture of experts,” inProceedings of the 60th Annual Meeting of the Association for Compu-tational Linguistics (Volume 1: Long Papers), 2022, pp. 7085–7095.





[96] T. Chen, Z. Zhang, A. K. JAISWAL, S. Liu, and Z. Wang, “Sparsemoe as the new dropout: Scaling dense and self-slimmabletransformers,” in The Eleventh International Conference on LearningRepresentations, 2022.





[97] N. Du, Y. Huang, A. M. Dai, S. Tong, D. Lepikhin, Y. Xu,M. Krikun, Y. Zhou, A. W. Yu, O. Firat et al., “Glam: Efficient scal-ing of language models with mixture-of-experts,” in InternationalConference on Machine Learning. PMLR, 2022, pp. 5547–5569.





[98] N. Shazeer, A. Mirhoseini, K. Maziarz, A. Davis, Q. Le, G. Hinton,and J. Dean, “Outrageously large neural networks: The sparsely-gated mixture-of-experts layer,” in International Conference onLearning Representations, 2016.





[99] D. Lepikhin, H. Lee, Y. Xu, D. Chen, O. Firat, Y. Huang,M. Krikun, N. Shazeer, and Z. Chen, “Gshard: Scaling giantmodels with conditional computation and automatic sharding,”arXiv preprint arXiv:2006.16668, 2020.





[100] C. Hwang, W. Cui, Y. Xiong, Z. Yang, Z. Liu, H. Hu, Z. Wang,R. Salas, J. Jose, P. Ram et al., “Tutel: Adaptive mixture-of-expertsat scale,” Proceedings of Machine Learning and Systems, vol. 5, 2023.





[101] D. P. Bertsekas, “Auction algorithms for network flow problems:A tutorial introduction,” Computational optimization and applica-tions, vol. 1, pp. 7–66, 1992.





[102] Z. Dai, G. Lai, Y. Yang, and Q. Le, “Funnel-transformer: Filteringout sequential redundancy for efficient language processing,”Advances in neural information processing systems, vol. 33, pp. 4271–4282, 2020.





[103] L. Zhu, B. Liao, Q. Zhang, X. Wang, W. Liu, and X. Wang,“Vision mamba: Efficient visual representation learning withbidirectional state space model,” arXiv preprint arXiv:2401.09417,2024.





[104] W. Hua, Z. Dai, H. Liu, and Q. Le, “Transformer quality in lineartime,” in International Conference on Machine Learning. PMLR,2022, pp. 9099–9117.





[105] AI21, “Jamba: Ai21’s groundbreaking ssm-transformer model,”March 2024. [Online]. Available: https://www.ai21.com/blog/announcing-jamba





[106] W. He, K. Han, Y. Tang, C. Wang, Y. Yang, T. Guo, andY. Wang, “Densemamba: State space models with dense hiddenconnection for efficient large language models,” arXiv preprintarXiv:2403.00818, 2024.





[107] Q. Anthony, Y. Tokpanov, P. Glorioso, and B. Millidge, “Black-mamba: Mixture of experts for state-space models,” arXiv preprintarXiv:2402.01771, 2024.





[108] M. Pioro, K. Ciebiera, K. Kr ´ ol, J. Ludziejewski, and S. Jaszczur, ´“Moe-mamba: Efficient selective state space models with mixtureof experts,” arXiv preprint arXiv:2401.04081, 2024.





[109] S. Zhai, W. Talbott, N. Srivastava, C. Huang, H. Goh, R. Zhang,and J. Susskind, “An attention free transformer,” arXiv preprintarXiv:2105.14103, 2021.





[110] T. Schuster, A. Fisch, J. Gupta, M. Dehghani, D. Bahri, V. Tran,Y. Tay, and D. Metzler, “Confident adaptive language modeling,”Advances in Neural Information Processing Systems, vol. 35, pp.17 456–17 472, 2022.





[111] L. Del Corro, A. Del Giorno, S. Agarwal, B. Yu, A. Awadallah, andS. Mukherjee, “Skipdecode: Autoregressive skip decoding withbatching and caching for efficient llm inference,” arXiv preprintarXiv:2307.02628, 2023.





[112] W. Liu, P. Zhou, Z. Wang, Z. Zhao, H. Deng, and Q. Ju, “Fastbert:a self-distilling bert with adaptive inference time,” in Proceedingsof the 58th Annual Meeting of the Association for ComputationalLinguistics, 2020, pp. 6035–6044.





[113] J. Kong, J. Wang, L.-C. Yu, and X. Zhang, “Accelerating inferencefor pretrained language models by unified multi-perspectiveearly exiting,” in Proceedings of the 29th International Conferenceon Computational Linguistics, 2022, pp. 4677–4686.





[114] K. Liao, Y. Zhang, X. Ren, Q. Su, X. Sun, and B. He, “A globalpast-future early exit method for accelerating inference of pre-trained language models,” in Proceedings of the 2021 Conferenceof the North American Chapter of the Association for ComputationalLinguistics: Human Language Technologies, 2021, pp. 2013–2023.





[115] J. Xin, R. Tang, J. Lee, Y. Yu, and J. Lin, “Deebert: Dynamicearly exiting for accelerating bert inference,” in Proceedings of the58th Annual Meeting of the Association for Computational Linguistics,2020, pp. 2246–2251.





[116] W. Zhou, C. Xu, T. Ge, J. McAuley, K. Xu, and F. Wei, “Bert losespatience: Fast and robust inference with early exit,” Advances inNeural Information Processing Systems, vol. 33, pp. 18 330–18 341,2020.





[117] T. Sun, X. Liu, W. Zhu, Z. Geng, L. Wu, Y. He, Y. Ni, G. Xie, X.-J.Huang, and X. Qiu, “A simple hash-based early exiting approachfor language understanding and generation,” in Findings of theAssociation for Computational Linguistics: ACL 2022, 2022, pp. 2409–2421.





[118] Y. Huang, Y. Chen, Z. Yu, and K. McKeown, “In-context learningdistillation: Transferring few-shot learning ability of pre-trainedlanguage models,” arXiv preprint arXiv:2212.10670, 2022.





[119] J. Zhao, W. Zhao, A. Drozdov, B. Rozonoyer, M. A. Sultan,J.-Y. Lee, M. Iyyer, and A. McCallum, “Multistage collabora-tive knowledge distillation from large language models,” arXivpreprint arXiv:2311.08640, 2023.





[120] C.-Y. Hsieh, C.-L. Li, C.-K. Yeh, H. Nakhost, Y. Fujii, A. Ratner,R. Krishna, C.-Y. Lee, and T. Pfister, “Distilling step-by-step!





outperforming larger language models with less training dataand smaller model sizes,” arXiv preprint arXiv:2305.02301, 2023.





[121] L. H. Li, J. Hessel, Y. Yu, X. Ren, K.-W. Chang, and Y. Choi,“Symbolic chain-of-thought distillation: Small models can also”think” step-by-step,” arXiv preprint arXiv:2306.14050, 2023.





[122] L. C. Magister, J. Mallinson, J. Adamek, E. Malmi, and A. Sev-eryn, “Teaching small language models to reason,” arXiv preprintarXiv:2212.08410, 2022.





[123] H. Chen, S. Wu, X. Quan, R. Wang, M. Yan, and J. Zhang, “Mcc-kd: Multi-cot consistent knowledge distillation,” arXiv preprintarXiv:2310.14747, 2023.





[124] N. Ho, L. Schmid, and S.-Y. Yun, “Large language models arereasoning teachers,” arXiv preprint arXiv:2212.10071, 2022.





[125] K. Shridhar, A. Stolfo, and M. Sachan, “Distilling reasoningcapabilities into smaller language models,” in Findings of theAssociation for Computational Linguistics: ACL 2023, 2023, pp. 7059–7073.





[126] X. Zhu, B. Qi, K. Zhang, X. Long, and B. Zhou, “Pad: Program-aided distillation specializes large models in reasoning,” arXivpreprint arXiv:2305.13888, 2023.





[127] P. Wang, Z. Wang, Z. Li, Y. Gao, B. Yin, and X. Ren, “Scott:Self-consistent chain-of-thought distillation,” arXiv preprintarXiv:2305.01879, 2023.





[128] Z. Chen, Q. Gao, A. Bosselut, A. Sabharwal, and K. Richardson,“Disco: distilling counterfactuals with large language models,”in Proceedings of the 61st Annual Meeting of the Association forComputational Linguistics (Volume 1: Long Papers), 2023, pp. 5514–5528.





[129] M. Wu, A. Waheed, C. Zhang, M. Abdul-Mageed, and A. F. Aji,“Lamini-lm: A diverse herd of distilled models from large-scaleinstructions,” arXiv preprint arXiv:2304.14402, 2023.





[130] Y. Jiang, C. Chan, M. Chen, and W. Wang, “Lion: Adversarialdistillation of proprietary large language models,” in Proceedingsof the 2023 Conference on Empirical Methods in Natural LanguageProcessing, 2023, pp. 3134–3154.





[131] Y. Gu, L. Dong, F. Wei, and M. Huang, “Knowledge distillationof large language models,” arXiv preprint arXiv:2306.08543, 2023.





[132] R. Agarwal, N. Vieillard, P. Stanczyk, S. Ramos, M. Geist, andO. Bachem, “Gkd: Generalized knowledge distillation for auto-regressive sequence models,” arXiv preprint arXiv:2306.13649,2023.





[133] C. Liang, S. Zuo, Q. Zhang, P. He, W. Chen, and T. Zhao, “Lessis more: Task-aware layer-wise distillation for language modelcompression,” in International Conference on Machine Learning.PMLR, 2023, pp. 20 852–20 867.





[134] I. Timiryasov and J.-L. Tastet, “Baby llama: knowledge distillationfrom an ensemble of teachers trained on a small dataset with noperformance penalty,” arXiv preprint arXiv:2308.02019, 2023.





[135] C. Zhang, Y. Yang, J. Liu, J. Wang, Y. Xian, B. Wang, and D. Song,“Lifting the curse of capacity gap in distilling language models,”arXiv preprint arXiv:2305.12129, 2023.





[136] L. Hou, Z. Huang, L. Shang, X. Jiang, X. Chen, and Q. Liu, “Dyn-abert: Dynamic bert with adaptive width and depth,” Advancesin Neural Information Processing Systems, vol. 33, pp. 9782–9793,2020.





[137] S. Padmanabhan, Y. Onoe, M. J. Zhang, G. Durrett, and E. Choi,“Propagating knowledge updates to lms through distillation,”arXiv preprint arXiv:2306.09306, 2023.





[138] Y. Yin, C. Chen, L. Shang, X. Jiang, X. Chen, and Q. Liu, “Au-totinybert: Automatic hyper-parameter optimization for efficientpre-trained language models,” arXiv preprint arXiv:2107.13686,2021.





[139] J. Xu, X. Tan, R. Luo, K. Song, J. Li, T. Qin, and T.-Y. Liu, “Nas-bert: task-agnostic and adaptive-size bert compression with neu-ral architecture search,” in Proceedings of the 27th ACM SIGKDDConference on Knowledge Discovery & Data Mining, 2021, pp. 1933–1943.





[140] A. Klein, J. Golebiowski, X. Ma, V. Perrone, and C. Archambeau,“Structural pruning of large language models via neural archi-tecture search,” 2023.





[141] M. Javaheripi, G. de Rosa, S. Mukherjee, S. Shah, T. Religa,C. C. Teodoro Mendes, S. Bubeck, F. Koushanfar, and D. Dey,“Litetransformersearch: Training-free neural architecture searchfor efficient language models,” Advances in Neural InformationProcessing Systems, vol. 35, pp. 24 254–24 267, 2022.





[142] D. D. Xu, S. Mukherjee, X. Liu, D. Dey, W. Wang, X. Zhang,A. Awadallah, and J. Gao, “Few-shot task-agnostic neural archi-





tecture search for distilling large language models,” Advances inNeural Information Processing Systems, vol. 35, pp. 28 644–28 656,2022.





[143] A. Kaushal, T. Vaidhya, and I. Rish, “Lord: Low rank decomposi-tion of monolingual code llms for one-shot compression,” arXivpreprint arXiv:2309.14021, 2023.





[144] M. Xu, Y. L. Xu, and D. P. Mandic, “Tensorgpt: Efficient com-pression of the embedding layer in llms based on the tensor-traindecomposition,” arXiv preprint arXiv:2307.00526, 2023.





[145] Y. Li, Y. Yu, Q. Zhang, C. Liang, P. He, W. Chen, and T. Zhao,“Losparse: Structured compression of large language modelsbased on low-rank and sparse approximation,” arXiv preprintarXiv:2306.11222, 2023.





[146] R. Saha, V. Srivastava, and M. Pilanci, “Matrix compression viarandomized low rank and low precision factorization,” arXivpreprint arXiv:2310.11028, 2023.





[147] Z. Yao, X. Wu, C. Li, S. Youn, and Y. He, “Zeroquant-v2: Exploringpost-training quantization in llms from comprehensive study tolow rank compensation,” arXiv preprint arXiv:2303.08302, 2023.





[148] R. Chand, Y. Prabhu, and P. Kumar, “Dsformer: Effective com-pression of text-transformers by dense-sparse weight factoriza-tion,” arXiv preprint arXiv:2312.13211, 2023.





[149] Z. Yuan, Y. Shang, Y. Song, Q. Wu, Y. Yan, and G. Sun, “Asvd:Activation-aware singular value decomposition for compressinglarge language models,” arXiv preprint arXiv:2312.05821, 2023.





[150] R. Child, S. Gray, A. Radford, and I. Sutskever, “Generat-ing long sequences with sparse transformers,” arXiv preprintarXiv:1904.10509, 2019.





[151] G. Xiao, Y. Tian, B. Chen, S. Han, and M. Lewis, “Efficientstreaming language models with attention sinks,” arXiv preprintarXiv:2309.17453, 2023.





[152] I. Beltagy, M. E. Peters, and A. Cohan, “Longformer: The long-document transformer,” arXiv preprint arXiv:2004.05150, 2020.





[153] M. Zaheer, G. Guruganesh, K. A. Dubey, J. Ainslie, C. Alberti,S. Ontanon, P. Pham, A. Ravula, Q. Wang, L. Yang et al., “Bigbird: Transformers for longer sequences,” Advances in neuralinformation processing systems, vol. 33, pp. 17 283–17 297, 2020.





[154] S. Dai, H. Genc, R. Venkatesan, and B. Khailany, “Efficient trans-former inference with statically structured sparse attention,” in2023 60th ACM/IEEE Design Automation Conference (DAC). IEEE,2023, pp. 1–6.





[155] Anonymous, “SemSA: Semantic sparse attention is hiddenin large language models.” 2023. [Online]. Available: https://openreview.net/forum?id=eG9AkHtYYH





[156] H. Wang, Z. Zhang, and S. Han, “Spatten: Efficient sparse at-tention architecture with cascade token and head pruning,” in2021 IEEE International Symposium on High-Performance ComputerArchitecture (HPCA). IEEE, pp. 97–110.





[157] L. Ren, Y. Liu, S. Wang, Y. Xu, C. Zhu, and C. Zhai, “Sparse mod-ular activation for efficient sequence modeling,” arXiv preprintarXiv:2306.11197, 2023.





[158] S. Anagnostidis, D. Pavllo, L. Biggio, L. Noci, A. Lucchi,and T. Hoffmann, “Dynamic context pruning for efficientand interpretable autoregressive transformers,” arXiv preprintarXiv:2305.15805, 2023.





[159] N. Kitaev, Ł. Kaiser, and A. Levskaya, “Reformer: The efficienttransformer,” arXiv preprint arXiv:2001.04451, 2020.





[160] M. Pagliardini, D. Paliotta, M. Jaggi, and F. Fleuret, “Faster causalattention over large sequences through sparse flash attention,”arXiv preprint arXiv:2306.01160, 2023.





[161] A. Roy, M. Saffar, A. Vaswani, and D. Grangier, “Efficient content-based sparse attention with routing transformers,” Transactions ofthe Association for Computational Linguistics, vol. 9, pp. 53–68, 2021.





[162] Y. Tay, D. Bahri, L. Yang, D. Metzler, and D.-C. Juan, “Sparsesinkhorn attention,” in International Conference on Machine Learn-ing. PMLR, 2020, pp. 9438–9447.





[163] Z. Zhang, Y. Sheng, T. Zhou, T. Chen, L. Zheng, R. Cai, Z. Song,Y. Tian, C. Re, C. Barrett ´ et al., “H2o: Heavy-hitter oracle forefficient generative inference of large language models,” Advancesin Neural Information Processing Systems, vol. 36, 2024.





[164] A. Feng, I. Li, Y. Jiang, and R. Ying, “Diffuser: efficient transform-ers with multi-hop attention diffusion for long sequences,” inProceedings of the AAAI Conference on Artificial Intelligence, vol. 37,no. 11, 2023, pp. 12 772–12 780.





[165] E. Frantar and D. Alistarh, “Sparsegpt: Massive language modelscan be accurately pruned in one-shot,” 2023.





[166] M. Sun, Z. Liu, A. Bair, and J. Z. Kolter, “A simple and effectivepruning approach for large language models,” arXiv preprintarXiv:2306.11695, 2023.





[167] H. Shao, B. Liu, and Y. Qian, “One-shot sensitivity-aware mixedsparsity pruning for large language models,” arXiv preprintarXiv:2310.09499, 2023.





[168] A. Syed, P. H. Guo, and V. Sundarapandiyan, “Prune and tune:Improving efficient pruning techniques for massive languagemodels,” 2023.





[169] X. Wei, Y. Zhang, Y. Li, X. Zhang, R. Gong, J. Guo, and X. Liu,“Outlier suppression+: Accurate quantization of large languagemodels by equivalent and optimal shifting and scaling,” arXivpreprint arXiv:2304.09145, 2023.





[170] P. Xu, W. Shao, M. Chen, S. Tang, K. Zhang, P. Gao, F. An,Y. Qiao, and P. Luo, “Besa: Pruning large language models withblockwise parameter-efficient sparsity allocation,” in The TwelfthInternational Conference on Learning Representations, 2023.





[171] E. Kurtic, D. Campos, T. Nguyen, E. Frantar, M. Kurtz, B. Fineran,M. Goin, and D. Alistarh, “The optimal bert surgeon: Scalableand accurate second-order pruning for large language models,”in Proceedings of the 2022 Conference on Empirical Methods in NaturalLanguage Processing, 2022, pp. 4163–4181.





[172] W. Kwon, S. Kim, M. W. Mahoney, J. Hassoun, K. Keutzer,and A. Gholami, “A fast post-training pruning framework fortransformers,” Advances in Neural Information Processing Systems,vol. 35, pp. 24 101–24 116, 2022.





[173] Y. Zhang, H. Bai, H. Lin, J. Zhao, L. Hou, and C. V. Cannistraci,“An efficient plug-and-play post-training pruning strategy inlarge language models,” 2023.





[174] X. Ma, G. Fang, and X. Wang, “Llm-pruner: On the structuralpruning of large language models,” Advances in neural informationprocessing systems, vol. 36, 2024.





[175] M. Xia, T. Gao, Z. Zeng, and D. Chen, “Sheared llama: Accelerat-ing language model pre-training via structured pruning,” arXivpreprint arXiv:2310.06694, 2023.





[176] E. Kurtic, E. Frantar, and D. Alistarh, “Ziplm: Inference-aware ´structured pruning of language models,” Advances in NeuralInformation Processing Systems, vol. 36, 2024.





[177] M. Zhang, H. Chen, C. Shen, Z. Yang, L. Ou, X. Yu, andB. Zhuang, “Loraprune: Pruning meets low-rank parameter-efficient fine-tuning,” 2023.





[178] T. Chen, T. Ding, B. Yadav, I. Zharkov, and L. Liang, “Lorashear:Efficient large language model structured pruning and knowl-edge recovery,” arXiv preprint arXiv:2310.18356, 2023.





[179] S. Ashkboos, M. L. Croci, M. G. d. Nascimento, T. Hoefler,and J. Hensman, “Slicegpt: Compress large language modelsby deleting rows and columns,” arXiv preprint arXiv:2401.15024,2024.





[180] Q. Zhang, S. Zuo, C. Liang, A. Bukharin, P. He, W. Chen,and T. Zhao, “Platon: Pruning large transformer models withupper confidence bound of weight importance,” in InternationalConference on Machine Learning. PMLR, 2022, pp. 26 809–26 823.





[181] M. Xia, Z. Zhong, and D. Chen, “Structured pruning learnscompact and accurate models,” in Proceedings of the 60th AnnualMeeting of the Association for Computational Linguistics (Volume 1:Long Papers), 2022, pp. 1513–1528.





[182] C. Tao, L. Hou, H. Bai, J. Wei, X. Jiang, Q. Liu, P. Luo, andN. Wong, “Structured pruning for efficient generative pre-trainedlanguage models,” in Findings of the Association for ComputationalLinguistics: ACL 2023, 2023, pp. 10 880–10 895.





[183] X. Lu, Q. Liu, Y. Xu, A. Zhou, S. Huang, B. Zhang, J. Yan, andH. Li, “Not all experts are equal: Efficient expert pruning andskipping for mixture-of-experts large language models,” arXivpreprint arXiv:2402.14800, 2024.





[184] A. Muzio, A. Sun, and C. He, “Seer-moe: Sparse expert efficiencythrough regularization for mixture-of-experts,” arXiv preprintarXiv:2404.05089, 2024.





[185] P. Dong, L. Li, Z. Tang, X. Liu, X. Pan, Q. Wang, and X. Chu,“Pruner-zero: Evolving symbolic pruning metric from scratch forlarge language models,” in International Conference on MachineLearning (ICML), 2024.





[186] Y. Zhang, L. Zhao, M. Lin, S. Yunyun, Y. Yao, X. Han, J. Tanner,S. Liu, and R. Ji, “Dynamic sparse no training: Training-free fine-tuning for sparse llms,” in International Conference on LearningRepresentations (ICLR), 2024.





[187] S.-y. Liu, Z. Liu, X. Huang, P. Dong, and K.-T. Cheng, “Llm-fp4: 4-bit floating-point quantized transformers,” in The 2023 Conferenceon Empirical Methods in Natural Language Processing, 2023.





[188] L. Li, Q. Li, B. Zhang, and X. Chu, “Norm tweaking: High-performance low-bit quantization of large language models,”arXiv preprint arXiv:2309.02784, 2023.





[189] T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer,“Qlora: Efficient finetuning of quantized llms,” Advances in Neu-ral Information Processing Systems, vol. 36, 2024.





[190] Y. Xu, L. Xie, X. Gu, X. Chen, H. Chang, H. Zhang, Z. Chen,X. Zhang, and Q. Tian, “Qa-lora: Quantization-aware low-rank adaptation of large language models,” arXiv preprintarXiv:2309.14717, 2023.





[191] Y. Li, Y. Yu, C. Liang, P. He, N. Karampatziakis, W. Chen, andT. Zhao, “Loftq: Lora-fine-tuning-aware quantization for largelanguage models,” arXiv preprint arXiv:2310.08659, 2023.





[192] E. Frantar, S. Ashkboos, T. Hoefler, and D. Alistarh, “Gptq:Accurate post-training quantization for generative pre-trainedtransformers,” arXiv preprint arXiv:2210.17323, 2022.





[193] G. Park, M. Kim, S. Lee, J. Kim, B. Kwon, S. J. Kwon, B. Kim,Y. Lee, D. Lee et al., “Lut-gemm: Quantized matrix multiplicationbased on luts for efficient inference in large-scale generative lan-guage models,” in The Twelfth International Conference on LearningRepresentations, 2023.





[194] J. Lin, J. Tang, H. Tang, S. Yang, X. Dang, and S. Han, “Awq:Activation-aware weight quantization for llm compression andacceleration,” arXiv preprint arXiv:2306.00978, 2023.





[195] C. Lee, J. Jin, T. Kim, H. Kim, and E. Park, “Owq: Lessons learnedfrom activation outliers for weight quantization in large languagemodels,” arXiv preprint arXiv:2306.02272, 2023.





[196] T. Dettmers, R. Svirschevski, V. Egiazarian, D. Kuznedelev,E. Frantar, S. Ashkboos, A. Borzunov, T. Hoefler, and D. Alistarh,“Spqr: A sparse-quantized representation for near-lossless llmweight compression,” arXiv preprint arXiv:2306.03078, 2023.





[197] S. Kim, C. Hooper, A. Gholami, Z. Dong, X. Li, S. Shen, M. W.Mahoney, and K. Keutzer, “Squeezellm: Dense-and-sparse quan-tization,” arXiv preprint arXiv:2306.07629, 2023.





[198] J. Chee, Y. Cai, V. Kuleshov, and C. De Sa, “Quip: 2-bit quantiza-tion of large language models with guarantees,” in Thirty-seventhConference on Neural Information Processing Systems, 2023.





[199] Y. J. Kim, R. Henry, R. Fahim, and H. H. Awadalla, “Finequant:Unlocking efficiency with fine-grained weight-only quantizationfor llms,” arXiv preprint arXiv:2308.09723, 2023.





[200] K. Behdin, A. Acharya, A. Gupta, S. Keerthi, and R. Mazumder,“Quantease: Optimization-based quantization for languagemodels–an efficient and intuitive algorithm,” arXiv preprintarXiv:2309.01885, 2023.





[201] S. Li, X. Ning, K. Hong, T. Liu, L. Wang, X. Li, K. Zhong, G. Dai,H. Yang, and Y. Wang, “Llm-mq: Mixed-precision quantizationfor efficient llm deployment,” 2023.





[202] Z. Yao, R. Y. Aminabadi, M. Zhang, X. Wu, C. Li, and Y. He,“Zeroquant: Efficient and affordable post-training quantizationfor large-scale transformers,” in Advances in Neural InformationProcessing Systems, 2022.





[203] Y. Sheng, L. Zheng, B. Yuan, Z. Li, M. Ryabinin, B. Chen, P. Liang,C. Re, I. Stoica, and C. Zhang, “Flexgen: High-throughput gen-erative inference of large language models with a single gpu,”2023.





[204] T. Dettmers, M. Lewis, Y. Belkada, and L. Zettlemoyer, “Llm. int8(): 8-bit matrix multiplication for transformers at scale,” arXivpreprint arXiv:2208.07339, 2022.





[205] G. Xiao, J. Lin, M. Seznec, H. Wu, J. Demouth, and S. Han,“Smoothquant: Accurate and efficient post-training quantizationfor large language models,” in International Conference on MachineLearning. PMLR, 2023, pp. 38 087–38 099.





[206] Z. Yao, X. Wu, C. Li, S. Youn, and Y. He, “Zeroquant-v2: Exploringpost-training quantization in llms from comprehensive study tolow rank compensation,” arXiv preprint arXiv:2303.08302, 2023.





[207] Z. Yuan, L. Niu, J. Liu, W. Liu, X. Wang, Y. Shang, G. Sun, Q. Wu,J. Wu, and B. Wu, “Rptq: Reorder-based post-training quantiza-tion for large language models,” arXiv preprint arXiv:2304.01089,2023.





[208] C. Guo, J. Tang, W. Hu, J. Leng, C. Zhang, F. Yang, Y. Liu,M. Guo, and Y. Zhu, “Olive: Accelerating large language modelsvia hardware-friendly outlier-victim pair quantization,” in Pro-ceedings of the 50th Annual International Symposium on ComputerArchitecture, 2023, pp. 1–15.





[209] X. Wu, Z. Yao, and Y. He, “Zeroquant-fp: A leap forward in llmspost-training w4a8 quantization using floating-point formats,”arXiv preprint arXiv:2307.09782, 2023.





[210] W. Shao, M. Chen, Z. Zhang, P. Xu, L. Zhao, Z. Li, K. Zhang,P. Gao, Y. Qiao, and P. Luo, “Omniquant: Omnidirectionallycalibrated quantization for large language models,” in The TwelfthInternational Conference on Learning Representations, 2023.





[211] J. Liu, R. Gong, X. Wei, Z. Dong, J. Cai, and B. Zhuang, “Qllm:Accurate and efficient low-bitwidth quantization for large lan-guage models,” in The Twelfth International Conference on LearningRepresentations, 2023.





[212] Y. Zhao, C.-Y. Lin, K. Zhu, Z. Ye, L. Chen, S. Zheng, L. Ceze,A. Krishnamurthy, T. Chen, and B. Kasikci, “Atom: Low-bitquantization for efficient and accurate llm serving,” arXiv preprintarXiv:2310.19102, 2023.





[213] W. Huang, Y. Liu, H. Qin, Y. Li, S. Zhang, X. Liu, M. Magno, andX. Qi, “Billm: Pushing the limit of post-training quantization forllms,” 2024.





[214] S. Li, X. Ning, L. Wang, T. Liu, X. Shi, S. Yan, G. Dai, H. Yang, andY. Wang, “Evaluating quantized large language models,” arXivpreprint arXiv:2402.18158, 2024.





[215] Y. Ma, H. Li, X. Zheng, F. Ling, X. Xiao, R. Wang, S. Wen, F. Chao,and R. Ji, “Affinequant: Affine transformation quantization forlarge language models,” in International Conference on LearningRepresentations (ICLR), 2024.





[216] A. Tseng, J. Chee, Q. Sun, V. Kuleshov, and C. De Sa, “Quip#:Even better llm quantization with hadamard incoherence andlattice codebooks,” arXiv preprint arXiv:2402.04396, 2024.





[217] S. Ashkboos, A. Mohtashami, M. L. Croci, B. Li, M. Jaggi, D. Al-istarh, T. Hoefler, and J. Hensman, “Quarot: Outlier-free 4-bitinference in rotated llms,” arXiv preprint arXiv:2404.00456, 2024.





[218] Z. Liu, C. Zhao, I. Fedorov, B. Soran, D. Choudhary,R. Krishnamoorthi, V. Chandra, Y. Tian, and T. Blankevoort,“Spinquant–llm quantization with learned rotations,” arXivpreprint arXiv:2405.16406, 2024.





[219] C. Hooper, S. Kim, H. Mohammadzadeh, M. W. Mahoney, Y. S.Shao, K. Keutzer, and A. Gholami, “Kvquant: Towards 10 millioncontext length llm inference with kv cache quantization,” arXivpreprint arXiv:2401.18079, 2024.





[220] Z. Liu, J. Yuan, H. Jin, S. Zhong, Z. Xu, V. Braverman, B. Chen,and X. Hu, “Kivi: A tuning-free asymmetric 2bit quantization forkv cache,” arXiv preprint arXiv:2402.02750, 2024.





[221] E. Frantar and D. Alistarh, “Optimal brain compression: A frame-work for accurate post-training quantization and pruning,” inAdvances in Neural Information Processing Systems, 2022.





[222] N. Vaidya, F. Oh, and N. Comly, “Optimizing inference onlarge language models with nvidia tensorrt-llm, now pub-licly available,” [Online], 2023, https://github.com/NVIDIA/TensorRT-LLM.





[223] InternLM, “Lmdeploy,” 2024. [Online]. Available: https://github.com/InternLM/lmdeploy





[224] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang,and W. Chen, “Lora: Low-rank adaptation of large languagemodels,” arXiv preprint arXiv:2106.09685, 2021.





[225] B. Hassibi, D. G. Stork, and G. J. Wolff, “Optimal brain surgeonand general network pruning,” in IEEE international conference onneural networks. IEEE, 1993, pp. 293–299.





[226] Y. LeCun, J. Denker, and S. Solla, “Optimal brain damage,”Advances in neural information processing systems, vol. 2, 1989.





[227] D. C. Mocanu, E. Mocanu, P. Stone, P. H. Nguyen, M. Gibescu,and A. Liotta, “Scalable training of artificial neural networkswith adaptive sparse connectivity inspired by network science,”Nature communications, vol. 9, no. 1, p. 2383, 2018.





[228] B. Zoph and Q. Le, “Neural architecture search with reinforce-ment learning,” in International Conference on Learning Representa-tions, 2016.





[229] X. Wang, Y. Zheng, Z. Wan, and M. Zhang, “Svd-llm: Truncation-aware singular value decomposition for large language modelcompression,” arXiv preprint arXiv:2403.07378, 2024.





[230] L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. L. Wainwright,P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray et al., “Train-ing language models to follow instructions with human feedback,2022,” URL https://arxiv. org/abs/2203.02155, vol. 13, 2022.





[231] Y. Han, G. Huang, S. Song, L. Yang, H. Wang, and Y. Wang, “Dy-namic neural networks: A survey,” IEEE Transactions on PatternAnalysis and Machine Intelligence, vol. 44, no. 11, pp. 7436–7456,2021.





[232] C. Xu and J. McAuley, “A survey on dynamic neural networksfor natural language processing,” in Findings of the Association forComputational Linguistics: EACL 2023, 2023, pp. 2370–2381.





[233] X. He, I. Keivanloo, Y. Xu, X. He, B. Zeng, S. Rajagopalan, andT. Chilimbi, “Magic pyramid: Accelerating inference with earlyexiting and token pruning,” Image, 2023.





[234] TogetherAI, “Paving the way to efficient architectures:Stripedhyena-7b, open source models offering a glimpseinto a world beyond transformers,” December 2023. [Online].Available: https://www.together.ai/blog/stripedhyena-7b





[235] A. Jaiswal, Z. Gan, X. Du, B. Zhang, Z. Wang, and Y. Yang,“Compressing llms: The truth is rarely pure and never simple,”arXiv preprint arXiv:2310.01382, 2023.





[236] Y. Leviathan, M. Kalman, and Y. Matias, “Fast inference fromtransformers via speculative decoding,” in International Confer-ence on Machine Learning. PMLR, 2023, pp. 19 274–19 286.





[237] C. Chen, S. Borgeaud, G. Irving, J.-B. Lespiau, L. Sifre, andJ. Jumper, “Accelerating large language model decoding withspeculative sampling,” arXiv preprint arXiv:2302.01318, 2023.





[238] Y. Zhou, K. Lyu, A. S. Rawat, A. K. Menon, A. Rostamizadeh,S. Kumar, J.-F. Kagy, and R. Agarwal, “Distillspec: Improvingspeculative decoding via knowledge distillation,” arXiv preprintarXiv:2310.08461, 2023.





[239] J. Zhang, J. Wang, H. Li, L. Shou, K. Chen, G. Chen, and S. Mehro-tra, “Draft & verify: Lossless large language model accelerationvia self-speculative decoding,” arXiv preprint arXiv:2309.08168,2023.





[240] X. Liu, L. Hu, P. Bailis, I. Stoica, Z. Deng, A. Cheung,and H. Zhang, “Online speculative decoding,” arXiv preprintarXiv:2310.07177, 2023.





[241] G. Monea, A. Joulin, and E. Grave, “Pass: Parallel speculativesampling,” arXiv preprint arXiv:2311.13581, 2023.





[242] Z. He, Z. Zhong, T. Cai, J. D. Lee, and D. He, “Rest: Retrieval-based speculative decoding,” arXiv preprint arXiv:2311.08252,2023.





[243] X. Miao, G. Oliaro, Z. Zhang, X. Cheng, Z. Wang, R. Y. Y.Wong, Z. Chen, D. Arfeen, R. Abhyankar, and Z. Jia, “Specinfer:Accelerating generative llm serving with speculative inferenceand token tree verification,” arXiv preprint arXiv:2305.09781, 2023.





[244] B. Spector and C. Re, “Accelerating llm inference with stagedspeculative decoding,” arXiv preprint arXiv:2308.04623, 2023.





[245] Z. Chen, X. Yang, J. Lin, C. Sun, J. Huang, and K. C.-C. Chang,“Cascade speculative drafting for even faster llm inference,”arXiv preprint arXiv:2312.11462, 2023.





[246] Y. Fu, P. Bailis, I. Stoica, and H. Zhang, “Breaking the sequentialdependency of llm inference using lookahead decoding,”November 2023. [Online]. Available: https://lmsys.org/blog/2023-11-21-lookahead-decoding/





[247] Y. Li, C. Zhang, and H. Zhang, “Eagle: Lossless accelerationof llm decoding by feature extrapolation,” December 2023.[Online]. Available: https://sites.google.com/view/eagle-llm





[248] Z. Sun, A. T. Suresh, J. H. Ro, A. Beirami, H. Jain, and F. Yu,“Spectr: Fast speculative decoding via optimal transport,” arXivpreprint arXiv:2310.15141, 2023.





[249] F. Liu, Y. Tang, Z. Liu, Y. Ni, K. Han, and Y. Wang, “Kanga-roo: Lossless self-speculative decoding via double early exiting,”arXiv preprint arXiv:2404.18911, 2024.





[250] ggerganov, “Inference of meta’s llama model (and others) inpure $c / c / + + \prime \prime$ 2024. [Online]. Available: https://github.com/ggerganov/llama.cpp





[251] Y. Song, Z. Mi, H. Xie, and H. Chen, “Powerinfer: Fast largelanguage model serving with a consumer-grade gpu,” arXivpreprint arXiv:2312.12456, 2023.





[252] J. He and J. Zhai, “Fastdecode: High-throughput gpu-efficientllm serving using heterogeneous pipelines,” arXiv preprintarXiv:2403.11421, 2024.





[253] K. Hong, G. Dai, J. Xu, Q. Mao, X. Li, J. Liu, K. Chen, Y. Dong,and Y. Wang, “Flashdecoding++: Faster large language modelinference on gpus,” 2024.





[254] T. Gale, D. Narayanan, C. Young, and M. Zaharia, “Megablocks:Efficient sparse training with mixture-of-experts,” in Proceedingsof Machine Learning and Systems (MLSys), 2023.





[255] T. Dao, D. Fu, S. Ermon, A. Rudra, and C. Re, “Flashattention: ´Fast and memory-efficient exact attention with io-awareness,”Advances in Neural Information Processing Systems, vol. 35, pp.16 344–16 359, 2022.





[256] T. Dao, “Flashattention-2: Faster attention with better parallelismand work partitioning,” arXiv preprint arXiv:2307.08691, 2023.





[257] Y. Zhai, C. Jiang, L. Wang, X. Jia, S. Zhang, Z. Chen, X. Liu,and Y. Zhu, “Bytetransformer: A high-performance transformerboosted for variable-length inputs,” in 2023 IEEE InternationalParallel and Distributed Processing Symposium (IPDPS). IEEE,2023, pp. 344–355.





[258] R. Y. Aminabadi, S. Rajbhandari, A. A. Awan, C. Li, D. Li,E. Zheng, O. Ruwase, S. Smith, M. Zhang, J. Rasley et al.,“Deepspeed-inference: enabling efficient inference of transformermodels at unprecedented scale,” in SC22: International Conferencefor High Performance Computing, Networking, Storage and Analysis.IEEE, 2022, pp. 1–15.





[259] T. Dao, D. Haziza, F. Massa, and G. Sizov, “Flash-decoding forlong-context inference,” [Online], 2023, https://crfm.stanford.edu/2023/10/12/flashdecoding.html.





[260] HuggingFace, “Transformers: State-of-the-art machine learningfor pytorch, tensorflow, and jax.” [Online], 2024, https://github.com/huggingface/transformers.





[261] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux,T. Lacroix, B. Roziere, N. Goyal, E. Hambro, F. Azhar ` et al.,“Llama: Open and efficient foundation language models,” arXivpreprint arXiv:2302.13971, 2023.





[262] Z. Du, Y. Qian, X. Liu, M. Ding, J. Qiu, Z. Yang, and J. Tang,“Glm: General language model pretraining with autoregressiveblank infilling,” in Proceedings of the 60th Annual Meeting of theAssociation for Computational Linguistics (Volume 1: Long Papers),2022, pp. 320–335.





[263] Sensetime, “Openppl: A high-performance deep learning infer-ence platform,” [Online], 2023, https://openppl.ai/home.





[264] NVIDIA, “cublas: Basic linear algebra on nvidia gpus,” [Online],2017, https://developer.nvidia.com/cublas.





[265] ——, “Cutlass: Cuda templates for linear algebra subroutines,”[Online], 2017, https://github.com/NVIDIA/cutlass.





[266] S. Wang, “Fastgemv: High-speed gemv kernels,” [Online], 2023,https://github.com/wangsiping97/FastGEMV.





[267] P. Tillet, H. T. Kung, and D. Cox, “Triton: an intermediate lan-guage and compiler for tiled neural network computations,” inProceedings of the 3rd ACM SIGPLAN International Workshop onMachine Learning and Programming Languages, 2019, pp. 10–19.





[268] C. Zhang et al., “Beyond the speculative game: A survey ofspeculative execution in large language models,” arXiv preprintarXiv:2404.14897, 2024.





[269] H. Xia et al., “Unlocking efficiency in large language modelinference: A comprehensive survey of speculative decoding,”arXiv preprint arXiv:2401.07851, 2024.





[270] M. Stern, N. Shazeer, and J. Uszkoreit, “Blockwise paralleldecoding for deep autoregressive models,” Advances in NeuralInformation Processing Systems, vol. 31, 2018.





[271] V. Nair and G. E. Hinton, “Rectified linear units improve re-stricted boltzmann machines,” in Proceedings of the 27th interna-tional conference on machine learning (ICML-10), 2010, pp. 807–814.





[272] P. Patel, E. Choukse, C. Zhang, ´Inigo Goiri, A. Shah, S. Maleki, ˜and R. Bianchini, “Splitwise: Efficient generative llm inferenceusing phase splitting,” arXiv preprint arXiv:2311.18677, 2023.





[273] C. Hu, H. Huang, L. Xu, X. Chen, J. Xu, S. Chen, H. Feng,C. Wang, S. Wang, Y. Bao, N. Sun, and Y. Shan, “Inference withoutinterference: Disaggregate llm inference for mixed downstreamworkloads,” arXiv preprint arXiv:2401.11181, 2024.





[274] Y. Zhong, S. Liu, J. Chen, J. Hu, Y. Zhu, X. Liu, X. Jin, andH. Zhang, “Distserve: Disaggregating prefill and decoding forgoodput-optimized large language model serving,” arXiv preprintarXiv:2401.09670, 2024.





[275] X. Miao, C. Shi, J. Duan, X. Xi, D. Lin, B. Cui, and Z. Jia, “Spot-serve: Serving generative large language models on preemptibleinstances,” arXiv preprint arXiv:2311.15566, 2023.





[276] B. Lin, T. Peng, C. Zhang, M. Sun, L. Li, H. Zhao, W. Xiao,Q. Xu, X. Qiu, S. Li, Z. Ji, Y. Li, and W. Lin, “Infinite-llm: Efficientllm service for long context with distattention and distributedkvcache,” arXiv preprint arXiv:2401.02669, 2024.





[277] G.-I. Yu, J. S. Jeong, G.-W. Kim, S. Kim, and B.-G. Chun, “Orca:A distributed serving system for transformer-based generativemodels,” in Proceedings of the 16th USENIX Symposium on Operat-ing Systems Design and Implementation, 2022, pp. 521–538.





[278] ModelTC, “Lightllm,” February 2024. [Online]. Available:https://github.com/ModelTC/lightllm/





[279] C. Holmes, M. Tanaka, M. Wyatt, A. A. Awan, J. Rasley, S. Ra-jbhandari, R. Y. Aminabadi, H. Qin, A. Bakhtiari, L. Kurilenko,and Y. He, “Deepspeed-fastgen: High-throughput text genera-tion for llms via mii and deepspeed-inference,” arXiv preprintarXiv:2401.08671, 2024.





[280] B. Wu, Y. Zhong, Z. Zhang, G. Huang, X. Liu, and X. Jin, “Fastdistributed inference serving for large language models,” arXivpreprint arXiv:2305.05920, 2023.





[281] Y. Sheng, S. Cao, D. Li, B. Zhu, Z. Li, and D. Zhuo, “Fairness inserving large language models,” arXiv preprint arXiv:2401.00588,2024.





[282] A. Agrawal, A. Panwar, J. Mohan, N. Kwatra, B. Gulavani,and R. Ramjee, “Sarathi: Efficient llm inference by piggybackingdecodes with chunked prefills,” arXiv preprint arXiv:2308.16369,2023.





[283] A. Agrawal, N. Kedia, A. Panwar, J. Mohan, N. Kwatra, B. S.Gulavani, A. Tumanov, , and R. Ramjee, “Taming throughput-latency tradeoff in llm inference with sarathi-serve,” arXivpreprint arXiv:2403.02310, 2024.





[284] Y. Jin, C.-F. Wu, D. Brooks, and G.-Y. Wei, “S3: Increasing gpuutilization during generative inference for higher throughput,”arXiv preprint arXiv:2306.06000, 2023.





[285] Z. Ye, “flashinfer,” March 2024. [Online]. Available: https://github.com/flashinfer-ai/flashinfer





[286] NVIDIA, “Fastertransformer: About transformer related opti-mization, including bert, gpt,” [Online], 2017, https://github.com/NVIDIA/FasterTransformer.





[287] H. Oh, K. Kim, J. Kim, S. Kim, J. Lee, D.-s. Chang, and J. Seo,“Exegpt: Constraint-aware resource scheduling for llm infer-ence,” in Proceedings of the 29th ACM International Conference onArchitectural Support for Programming Languages and OperatingSystems, Volume 2, 2024, pp. 369–384.





[288] B. Sun, Z. Huang, H. Zhao, W. Xiao, X. Zhang, Y. Li, andW. Lin, “Llumnix: Dynamic scheduling for large language modelserving,” arXiv preprint arXiv:2406.03243, 2024.





[289] B. Wu, S. Liu, Y. Zhong, P. Sun, X. Liu, and X. Jin, “Loongserve:Efficiently serving long-context large language models with elas-tic sequence parallelism,” arXiv preprint arXiv:2404.09526, 2024.





[290] B. Li, S. Pandey, H. Fang, Y. Lyv, J. Li, J. Chen, M. Xie, L. Wan,H. Liu, and C. Ding, “Ftrans: Energy-efficient acceleration oftransformers using fpga,” arXiv preprint arXiv:2007.08563, 2020.





[291] T. J. Ham, Y. Lee, S. H. Seo, S. Kim, H. Choi, S. J. Jun, and J. W. Lee,“Elsa: Hardware-software co-design for efficient, lightweightself-attention mechanism in neural networks,” in ACM/IEEE 48thAnnual International Symposium on Computer Architecture, 2021,pp. 692–705.





[292] H. Fan, T. Chau, S. I. Venieris, R. Lee, A. Kouris, W. Luk, N. D.Lane, and M. S. Abdelfattah, “Adaptable butterfly accelerator forattention-based nns via hardware and algorithm co-design,” inIEEE/ACM International Symposium on Microarchitecture, 2022, pp.599–615.





[293] Y. Qin, Y. Wang, D. Deng, Z. Zhao, X. Yang, L. Liu, S. Wei,Y. Hu, and S. Yin, “Fact: Ffn-attention co-optimized transformerarchitecture with eager correlation prediction,” in Proceedings ofthe 50th Annual International Symposium on Computer Architecture,2023, pp. 1–14.





[294] H. Chen, J. Zhang, Y. Du, S. Xiang, Z. Yue, N. Zhang, Y. Cai,and Z. Zhang, “Understanding the potential of fpga-based spatialacceleration for large language model inference,” arXiv preprintarXiv:2312.15159, 2023.





[295] S. Hong, S. Moon, J. Kim, S. Lee, M. Kim, D. Lee, and J.-Y.Kim, “Dfx: A low-latency multi-fpga appliance for acceleratingtransformer-based text generation,” in IEEE Hot Chips 34 Sympo-sium, 2022.





[296] S. Zeng, J. Liu, G. Dai, X. Yang, T. Fu, H. Wang, W. Ma, H. Sun,S. Li, Z. Huang et al., “Flightllm: Efficient large language modelinference with a complete mapping flow on fpga,” arXiv preprintarXiv:2401.03868, 2024.





[297] S. teams, “Sharegpt,” 2023. [Online]. Available: https://sharegpt.com/





[298] J. Xie, Z. Chen, R. Zhang, X. Wan, and G. Li, “Large multimodalagents: A survey,” arXiv preprint arXiv:2402.15116, 2024.





[299] I. Lee, N. Jiang, and T. Berg-Kirkpatrick, “Exploring the relation-ship between model architecture and in-context learning ability,”arXiv preprint arXiv:2310.08049, 2023.





[300] S. Biderman, H. Schoelkopf, Q. G. Anthony, H. Bradley,K. O’Brien, E. Hallahan, M. A. Khan, S. Purohit, U. S. Prashanth,





E. Raff et al., “Pythia: A suite for analyzing large language modelsacross training and scaling,” in International Conference on MachineLearning. PMLR, 2023, pp. 2397–2430.





[301] J. Bai, S. Bai, Y. Chu, Z. Cui, K. Dang, X. Deng, Y. Fan, W. Ge,Y. Han, F. Huang et al., “Qwen technical report,” arXiv preprintarXiv:2309.16609, 2023.





[302] Y. Tang, F. Liu, Y. Ni, Y. Tian, Z. Bai, Y.-Q. Hu, S. Liu, S. Jui,K. Han, and Y. Wang, “Rethinking optimization and architecturefor tiny language models,” arXiv preprint arXiv:2402.02791, 2024.





[303] Y. Li, S. Bubeck, R. Eldan, A. Del Giorno, S. Gunasekar, and Y. T.Lee, “Textbooks are all you need ii: phi-1.5 technical report,”arXiv preprint arXiv:2309.05463, 2023.





[304] S. Gunasekar, Y. Zhang, J. Aneja, C. C. T. Mendes,A. Del Giorno, S. Gopi, M. Javaheripi, P. Kauffmann, G. de Rosa,O. Saarikivi et al., “Textbooks are all you need,” arXiv preprintarXiv:2306.11644, 2023.





[305] P. Zhang, G. Zeng, T. Wang, and W. Lu, “Tinyllama: An open-source small language model,” arXiv preprint arXiv:2401.02385,2024.





[306] C. Zhang, D. Song, Z. Ye, and Y. Gao, “Towards the lawof capacity gap in distilling language models,” arXiv preprintarXiv:2311.07052, 2023.





[307] X. Geng and H. Liu, “Openllama: An open reproduction ofllama,” May 2023. [Online]. Available: https://github.com/openlm-research/open llama





[308] M. Bellagente, J. Tow, D. Mahan, D. Phung, M. Zhuravinskyi,R. Adithyan, J. Baicoianu, B. Brooks, N. Cooper, A. Dattaet al., “Stable lm 2 1.6 b technical report,” arXiv preprintarXiv:2402.17834, 2024.





[309] “Minicpm: Unveiling the potential of end-side large languagemodels,” 2024.





[310] Z. Liu, C. Zhao, F. Iandola, C. Lai, Y. Tian, I. Fedorov, Y. Xiong,E. Chang, Y. Shi, R. Krishnamoorthi et al., “Mobilellm: Optimiz-ing sub-billion parameter language models for on-device usecases,” arXiv preprint arXiv:2402.14905, 2024.





[311] M. team, “MLC-LLM,” 2023. [Online]. Available: https://github.com/mlc-ai/mlc-llm





[312] Y. Yao, J. Duan, K. Xu, Y. Cai, Z. Sun, and Y. Zhang, “A survey onlarge language model (llm) security and privacy: The good, thebad, and the ugly,” High-Confidence Computing, p. 100211, 2024.





[313] Y. Li, H. Wen, W. Wang, X. Li, Y. Yuan, G. Liu, J. Liu, W. Xu,X. Wang, Y. Sun et al., “Personal llm agents: Insights and sur-vey about the capability, efficiency and security,” arXiv preprintarXiv:2401.05459, 2024.

