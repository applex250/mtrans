# 预训练基础模型全面综述：从 BERT 到 ChatGPT 的历史

Ce Zhou1* Qian Li2∗ Chen Li2∗ Jun Yu3∗ Yixin Liu3∗ Guangjing Wang1 Kai Zhang3 Cheng Ji2 Qiben Yan1 Lifang He3 Hao Peng2 Jianxin Li2 Jia Wu4 Ziwei Liu5 Pengtao Xie6 Caiming Xiong7 Jian Pei8 Philip S. Yu9 Lichao Sun3 

1密歇根州立大学， 2北京航空航天大学， 3利哈伊大学， 

4麦考瑞大学， 5南洋理工大学， 6加州大学圣地亚哥分校， 

7Salesforce AI Research， 8杜克大学， 9伊利诺伊大学芝加哥分校

# 摘要

预训练基础模型被视为各种不同数据模态下游任务的基础。PFM（例如 BERT、ChatGPT 和 GPT-4）是在大规模数据上训练的，这为广泛的下游应用提供了合理的参数初始化。与早期利用卷积和循环模块提取特征的方法不同，BERT 从 Transformer 中学习双向编码器表示，这些 Transformer 是在大规模数据集上作为上下文语言模型进行训练的。类似地，生成式预训练 Transformer (GPT) 方法采用 Transformer 作为特征提取器，并在大规模数据集上使用自回归范式进行训练。最近，ChatGPT 在大语言模型上显示了令人鼓舞的成功，它应用了带有零样本或少样本提示的自回归语言模型。PFM 的显著成就为近年来 AI 的各个领域带来了重大突破。许多研究提出了不同的方法、数据集和评估指标，这引发了对最新综述的需求。

本研究全面回顾了 PFM 在文本、图像、图以及其他数据模态方面的最新研究进展、挑战和机遇。该综述涵盖了自然语言处理、计算机视觉和图学习中使用的基本组件和现有的预训练方法。此外，它探讨了用于不同数据模态的高级 PFM，以及考虑数据质量和数量的统一 PFM。该综述还讨论了与 PFM 基础相关的研究，例如模型效率和压缩、安全性和隐私。最后，本研究提供了 PFM 领域的关键启示、未来研究方向、挑战和开放性问题。总体而言，本综述旨在阐明 PFM 在可扩展性、安全性、逻辑推理能力、跨域学习能力以及面向通用人工智能的用户友好交互能力方面的研究。

# 1 引言

预训练基础模型被视为大数据时代人工智能 (AI) 的核心和重要组成部分。基础模型这一名称首次出现在 [1] 中，它指的是一类更广泛的模型及其功能。PFMs 在三个主要的 AI 领域得到了广泛研究：自然语言处理 (NLP) [2]、计算机视觉 (CV) [3] 和图学习 (GL) [4]。PFMs 是功能强大的通用模型，在各种领域或跨领域都有效。它们在许多学习任务中展示了学习特征表示的巨大潜力，例如文本分类 [5]、文本生成 [6]、图像分类 [7]、目标检测 [8] 和图分类 [9]。PFMs 在使用大规模语料库训练多个任务并将其微调到类似的小规模任务方面表现出优越的性能，这使得快速启动数据处理成为可能。

# 1.1 PFMs 与预训练

PFMs 建立在预训练技术之上，其旨在使用大量数据和任务训练一个通用模型，该模型可以在不同的下游应用中轻松进行微调。预训练的思想源于 CV 任务中的迁移学习 [10]。认识到预训练在 CV 领域的有效性，人们开始使用预训练技术来提高其他领域的模型性能。当预训练技术应用于 NLP 领域时，训练良好的语言模型 可以捕获有利于下游任务的丰富知识，例如长期依赖关系、层次关系等。此外，预训练在 NLP 领域的一个显著优势是训练数据可以源自任何未标记的文本语料库，即在预训练过程中有无限量的训练数据。早期的预训练是一种静态技术，例如 NNLM [11] 和 Word2vec [12]，但静态方法难以适应不同的语义环境。因此，提出了动态预训练技术，例如 BERT [13]、XLNet [14] 等。图 1 描绘了 PFMs 在 NLP、CV 和 GL 领域的历史和演变。基于预训练技术的 PFMs 利用大型语料库来学习通用的语义表示。随着这些开创性工作的引入，各种 PFMs 涌现出来并被应用于下游任务和应用。

PFM 应用的一个很好的例子是 ChatGPT1。ChatGPT 是从生成式预训练 Transformer GPT-3.5 微调而来的，该模型是在文本和代码的混合数据上训练的 [15, 16]。ChatGPT 应用了基于人类反馈的强化学习 (RLHF) [17, 18]，这已成为使大语言模型 与人类意图保持一致的一种有前途的方式 [19]。ChatGPT 出奇优越的性能可能导致每种类型 PFM 训练范式发生转折点——应用指令对齐技术，例如强化学习 (RL)、提示微调 [20, 21, 22] 和思维链 [23, 24]，从而迈向通用人工智能。

我们重点回顾用于文本、图像和图的 PFMs，这是一个相对成熟的研究分类。对于文本，它是一种多用途 LM，用于预测序列中的下一个单词或字符。例如，PFMs 可用于机器翻译、问答系统、主题建模、情感分析等。对于图像，它类似于文本上的 PFMs，它利用海量数据集训练适合许多 CV 任务的大模型。对于图，类似的预训练思想也被应用来获取 PFMs，这些模型用于许多下游任务。除了针对特定数据领域的 PFMs 外，我们还回顾并陈述了一些其他的高级 PFMs，例如用于语音、视频和跨域数据的 PFMs，以及多模态 PFMs。一个典型的例子是 OpenAI [25] 描述的 GPT-4 模型，它是一个 massive multimodal language model（大型多模态语言模型），可以处理文本和图像输入并生成文本输出。GPT-4 在各种专业和学术评估任务中展示了人类水平的性能。此外，



![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-12/925beae9-6fc7-404b-a575-971981ce47ad/39a2454911b94a40172398b950475f2602eb53ebd6c7d2789eebd9bcd39d9e88.jpg)



图 1：PFMs 的历史与演进。

PFMs 中存在一种处理多模态数据的增长趋势，被称为统一 PFMs。该术语指的是可以处理不同类型数据（如文本、图像和音频）的模型。在这方面，我们给出了统一 PFMs 的定义，并回顾了近期研究中最先进的模型。值得注意的例子包括 OFA [26]、UNIFIED-IO [27]、FLAVA [28]、BEiT-3 [29] 等。

根据现有 PFMs 的特征，我们得出结论，PFMs 具有以下两个主要优势。首先，只需要很少的微调即可增强模型在下游任务上的性能。其次，PFMs 已经在质量方面得到了验证。我们可以应用 PFMs 到与任务相关的数据集，而不是从头开始构建模型来解决类似的问题。PFMs 的巨大前景激发了大量相关工作专注于模型效率 [30]、安全性 [31, 32, 33, 34] 和压缩 [35, 36]。

# 1.2 贡献与组织

有几项综述研究 [37, 8, 5, 6, 7, 1] 回顾了某些特定领域的预训练模型，例如文本生成 [6]、视觉 Transformer [7]、目标检测 [8]。

Bommasani 等人 [1] 总结了基础模型的机会和风险。然而，现有的工作并未对不同领域（例如 CV、NLP、GL、语音、视频）和不同方面（例如预训练任务、效率、有效性和隐私）的 PFMs 进行全面回顾。在本综述中，我们专门追踪 PFMs 在 NLP 领域的演变，以及预训练如何转移并被 CV 和 GL 采用。与其他综述相比，没有对来自这三个领域的现有 PFMs 进行全面的介绍和分析。与以前的预训练模型评论不同，我们总结了从传统模型到 PFMs 在这三个领域的近期工作的现有模型。传统模型强调静态特征学习。动态 PFMs 介绍了作为主流研究的结构。我们进一步介绍了 PFMs 的一些其他研究，包括其他高级和统一的 PFMs、模型效率和压缩、安全性和隐私。最后，我们总结了不同领域中未来的研究挑战和开放性问题。我们还在附录 F 和 G 中全面介绍了相关的评估指标和数据集。总之，主要贡献如下：

• 我们对 PFM 在 NLP、CV 和 GL 的发展进行了扎实且最新的回顾。在整个回顾过程中，我们讨论并提供了关于这三大应用领域之间通用 PFM 设计和预训练方法的见解。

• 我们总结了 PFMs 在其他多媒体领域（如语音和视频）的发展。此外，我们讨论了关于 PFMs 的高级主题，包括统一的 PFMs、模型效率和压缩以及安全性和隐私。

模型预训练：大数据集下的各种任务


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-12/925beae9-6fc7-404b-a575-971981ce47ad/ba040ed5f1c5bfe31f875116c3564ae1da46ca9060d71ac707dede77331a07c6.jpg)



图 2：PFMs 的一般概念架构：数据、模型和系统。

• 通过回顾用于不同任务的各种模态的 PFMs，我们讨论了大数据时代超大型模型未来研究的主要挑战和机遇，这为基于 PFM 的新一代协作和交互智能提供了指导。

本综述的其余部分组织如下。第 2 节介绍基本组件。第 3、4 和 5 节分别总结了 NLP、CV 和 GL 中的现有 PFMs。第 6、7 节介绍了 PFMs 的其他高级研究，包括高级和统一的 PFMs、模型效率和压缩以及安全性和隐私。此外，我们在第 9 节总结本综述之前，在第 8 节总结了 PFMs 的主要挑战。

# 2 基本组件

PFMs 的一般概念架构如图 2 所示。PFMs 是巨大的神经网络模型，它们都与神经信息处理有关。PFMs 的具体设计根据不同领域的数据模态和任务要求而有所不同。Transformer 是许多领域（如 NLP 和 CV）中 PFMs 的主流模型架构设计。训练大模型需要各种数据集来进行模型预训练。在训练 PFMs 后，应对模型进行微调以满足下游要求，例如有效性、效率和隐私。在本节中，我们介绍 NLP、CV 和 GL 领域中 PFMs 的基本模型架构、概念和设置。有关更详细组件的介绍，请参阅附录 A。

# 2.1 用于 PFMs 的 Transformer

Transformer [38] 是一种创新架构，有助于在各种神经单元之间转移加权表示知识。它完全依赖于注意力机制，不使用循环或卷积架构。注意力机制是 Transformer 的关键组成部分，因为它为所有编码的输入表示分配权重，并学习输入数据最重要的部分。注意力的输出是通过获取值的加权和获得的，权重是使用查询与相应键的兼容性函数计算的 [38]。在大模型中已经开发了许多注意力机制 [39]。例如，在自然语言处理中，创建了自注意力机制来连接单个序列中的各个位置，以生成同一序列的表示。Transformer 利用掩码矩阵提供基于自注意力的注意力机制，其中掩码矩阵指定哪些单词可以相互“看见”。

Transformer 是 NLP、CV 和 GL 领域中 PFMs 的重要结构。对于 NLP，Transformer 可以帮助解决处理序列输入数据时的长距离依赖问题。例如，GPT-3 [20] 是一种基于 Transformer 的生成模型。对于 CV，提出了视觉 Transformer (ViT) [40] 来将图像表示为一系列图像块，这类似于一系列词嵌入。对于 GL，采用图 Transformer 网络 (GTN) [41] 来学习新的图结构和强大的节点表示，而无需领域知识。由于 Transformer 结构实现了更高的并行化，Transformer 变得足够可扩展，从而为 PFMs 提供了突破性的能力。例如，ViT-22B 模型 [42] 大约有 22B 参数，最大的语言模型可以有超过 100B 参数（例如，GPT-3 有 175B，PaLM [43] 有 540B 参数）。

# 2.2 用于 PFMs 的学习机制

CV 中的深度学习模型在大多数任务中表现出比传统学习模型有很大的优势，包括常见的分类、识别、检测和分割任务以及特定的匹配、跟踪和序列预测。这些学习方法不仅适用于 CV，也适用于 NLP 和 GL。

监督学习 假设给定一个训练数据集 $\boldsymbol { X }$ 包含 $\{ ( \pmb { x } _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n }$ 来表示训练数据集中的原始数据，其中 $\mathbf { \delta } _ { \mathbf { \mathcal { X } } _ { i } }$ 表示第 $i$ 个训练样本，$y _ { i }$ 表示相应的标签。完整网络通过最小化目标函数来学习一个函数 $f ( { \pmb x } ; { \pmb \theta } )$，如下所示。

$$
\underset {\boldsymbol {\theta}} {\arg \min } \frac {1}{n} \sum_ {i = 1 } ^ {n} \mathcal {L} \left(f \left(\boldsymbol {x} _ {i}; \boldsymbol {\theta}\right), y _ {i}\right) + \lambda \Omega (\boldsymbol {\theta}), \tag {1}
$$

其中 $\mathcal { L }$ 和 $\Omega$ 分别代表预定义的损失函数和正则化项。函数 $f$ 具有嵌套形式，如

$$
\boldsymbol {h} _ {1} (\boldsymbol {x} _ {i}) = g \left(\boldsymbol {x} _ {i} ^ {\top} \boldsymbol {\omega} _ {1} + b _ {1}\right),
$$

$$
\boldsymbol {h} _ {l + 1} \left(\boldsymbol {x} _ {i}\right) = g \left(\boldsymbol {h} _ {l} \left(\boldsymbol {x} _ {i}\right) ^ {\top} \boldsymbol {\omega} _ {l} + b _ {l}\right), l = 1, 2, \dots , N \tag {2}
$$

其中 $l$ 是深度学习模型中层的索引，$N$ 是层数，这意味着 $\pmb { \theta } = \{ \omega _ { l } , b _ { l } , l = 1 , 2 , \cdots , N \}$ 。

半监督学习 假设除了之前带有人类标签的数据集外，给定另一个未标记数据集 ${ Z = \{ z _ { i } \} _ { i = 1 } ^ { m } }$。如果我们想利用这两个数据集来学习一个理想的网络，学习过程可以表述为

$$
\arg \min  _ {\boldsymbol {\theta}} \frac {1}{n} \sum_ {i = 1 } ^ {n} \mathcal {L} \left(f \left(\boldsymbol {x} _ {i}; \boldsymbol {\theta}\right), y _ {i}\right) + \frac {1}{m} \sum_ {i = 1 } ^ {m} \mathcal {L} ^ {\prime} \left(f ^ {\prime} \left(\boldsymbol {z} _ {i}; \boldsymbol {\theta} ^ {\prime}\right), R \left(\boldsymbol {z} _ {i}, \boldsymbol {X}\right)\right) + \lambda \Omega (\boldsymbol {\theta}), \tag {3}
$$

其中 $R$ 是一个关系函数，定义了未标记数据的目标，然后将这些伪标签集成到端到端的训练过程中。$f ^ { \prime }$ 是一个编码器，用于为数据集 $z$ 中的原始数据学习新的表示。具体来说，如果在训练过程中任何数据都没有标签，我们可以通过数据本身的内部距离或设计的代理任务来学习，这分别被称为无监督学习和自监督学习。后者是我们在第 4.3 节中详细讨论的主要内容。

弱监督学习 弱监督方法是根据对人类标签的依赖性在完全监督学习和 SSL 之间的平衡。SSL 设计特殊的代理任务来充当监督学习，而完全监督学习利用附加到数据上的现有标签。然而，它们都可以学习良好的视觉特征并在特定的下游任务上表现良好。假设数据集有 $K$ 个不准确的标签，并且任何标签都可以附加到数据样本上。因此，我们将图像 $\mathbf { \mathcal { x } } _ { i }$ 的真实标签表示为 $\pmb { y _ { i } } \in \{ 0 , 1 \} ^ { K } , i = 1 , 2 , \cdots , n$ ，并且 $\mathbf { \nabla } _ { \mathbf { \mathcal { Y } } _ { \lambda } }$ 的任何条目可以是 0 或 1。这里我们需要最小化总共 $n K$ 个损失项，公式如下。

$$
\underset {\boldsymbol {\theta}} {\arg \min } \frac {1}{n K} \sum_ {i = 1 } ^ {n} \sum_ {k = 1 } ^ {K} \mathcal {L} \left(f \left(\boldsymbol {x} _ {i}; \boldsymbol {\theta}\right), y _ {i } ^ {k}\right) + \lambda \Omega (\boldsymbol {\theta}), \tag {4}
$$

其中 $\left[ y _ { i } ^ { 1 } , y _ { i } ^ { 2 } , \cdot \cdot \cdot , y _ { i } ^ { K } \right] = y _ { i }$ ，并且 $\mathcal { L }$ 可以是适合二项式分类问题的损失函数。对于 $\mathbf { \nabla } _ { \mathbf { \mathcal { Y } } _ { \lambda } }$ 中的任何条目，需要计算一对多二项式分类的损失函数。

自监督学习 SSL 利用数据本身的信息来学习不同任务的基本特征表示。通过应用自定义的伪标签，它可以避免为 PFMs 手动标记大型数据集的成本。在 NLP 中，可以通过预测被掩码的字符、单词或句子来训练语言模型。变分自编码器 (VAE) 和生成对抗网络 (GAN) 是两种生成式 SSL 方法，它们用于重构数据本身。此外，对比学习作为一种判别式 SSL 方法，广泛应用于 CV、NLP 和 GL。对比学习的主要思想是借助数据增强等各种方法学习数据本身的先验知识分布。通过这种方式，对比学习可以学习一个模型，使相似实例在投影空间中更接近，不相似实例在投影空间中更远离。这里我们展示了一个简单的对比损失版本：

$$
\mathcal {L} _ {\mathrm {c}} \left(\mathbf {x} _ {i}, \mathbf {x} _ {j}, \theta\right) = m \| f _ {\theta} (\mathbf {x} _ {i}) - f _ {\theta} (\mathbf {x} _ {j}) \| _ {2} ^ {2} + (1 - m) \max  \left(0, \epsilon - \| f _ {\theta} (\mathbf {x} _ {i}) - f _ {\theta} (\mathbf {x} _ {j}) \| _ {2}\right) ^ {2} \tag {5}
$$

其中如果两个样本具有相同的标签，$m$ 为 1，否则为 0，$\epsilon$ 是距离上限。

强化学习 RL 是另一种学习范式，它将学习过程建模为智能体与环境之间的顺序交互，其中 RL 智能体寻求学习针对顺序决策问题的最优策略。具体来说，在每个时间交互步骤 $t$ ，智能体在状态空间 $s$ 中接收状态 $s _ { t }$ ，并遵循由 $\theta$ 参数化的策略 $\pi _ { \theta } ( a _ { t } | s _ { t } ) : { \mathcal { A } } \to S$ 从动作空间 $\mathcal { A }$ 中选择动作 $a _ { t }$ 。然后，智能体根据环境动力学接收标量即时奖励 $r _ { t } = r ( s _ { t } , a _ { t } )$ 和下一个状态 $s _ { t + 1 }$ ，其中 $r ( s , a )$ 是奖励函数。对于每个回合，此过程持续直到智能体达到终止状态。一个回合结束后，RL 智能体将重新开始一个新的回合。每个状态的回报是带有折扣因子 $\begin{array} { r } { \gamma \in ( 0 , 1 ] , R _ { t } = R ( s _ { t } , a _ { t } ) = \sum _ { k = 0 } ^ { \infty } \gamma ^ { k } r _ { t + k } } \end{array}$ 的折扣累积奖励。智能体旨在最大化每个状态这种长期回报的期望，

$$
\max  _ {\theta} \mathbb {E} _ {s _ {t}} [ R _ {t} | s _ {t}, a _ {t} = \pi_ {\theta} (s _ {t}) ]. \tag {6}
$$

# 2.3 用于 PFMs 的预训练任务

预训练是一个初始化框架，通常需要与下游任务的微调结合使用。在预训练和微调方案中，模型参数在预置任务上进行训练，以捕获特定属性、结构和社区信息。预训练特征可以辅助下游任务，提供充分信息，并加快模型收敛速度。

# 2.3.1 用于 NLP 的预训练任务

根据学习方法，预训练任务可以分为五类：掩码语言建模 (MLM)、去噪自编码器 (DAE)、替换标记检测 (RTD)、下一句预测 (NSP)、句子顺序预测 (SOP)。RTD、NSP 和 SOP 是对比学习方法，它们假设观察到的样本比随机样本在语义上更相似。

掩码语言建模 (MLM)。MLM 随机擦除输入序列中的一些单词，然后在预训练期间预测这些被擦除的单词。典型例子包括 BERT [13] 和 SpanBERT [44]。

去噪自编码器 (DAE)。DAE 用于向原始语料库添加噪声，并使用包含噪声的语料库重构原始输入。BART [45] 是一个代表性例子。

替换标记检测 (RTD)。RTD 是一个判别任务，用于确定 LM 是否替换了当前标记。该任务在 ELECTRA [46] 中引入。通过训练模型来区分标记是否被替换，模型可以获取语言知识。

下一句预测 (NSP)。为了使模型理解两个句子之间的相关性并捕获句子级表示，引入了 NSP 任务。PFM 输入来自不同文档的两个句子，并检查句子的顺序是否正确。一个典型的例子是 BERT。

句子顺序预测 (SOP)。与 NSP 不同，SOP 使用文档中的两个连续片段作为正样本，并交换两个片段的顺序作为负样本。PFMs 可以更好地建模句子之间的相关性，例如 ALBERT [47]。

# 2.3.2 用于 CV 的预训练任务

有许多为 CV 创建的预训练任务用于学习特征空间，这是基于 SSL 的。它利用包含人类设计标签的代理任务，例如拼图游戏或图像中各种补丁的比较。这使得学习到的表示能够泛化到一系列下游任务。

特定代理任务。代理任务也称为预定义任务，是供编码器网络在预训练阶段执行的。网络通过预测特殊代理任务的答案来进行训练。基于数据的特定特征，为虚拟任务生成伪标签。然后，使用引导学习技术训练编码器网络来解决代理任务。例如，图像修复旨在通过预测缺失的中心部分来预训练模型。

帧顺序学习任务。从视频中学习帧顺序涉及通过时间步进行帧处理，这可以作为 CV 的预训练任务。这个问题通常与完成有助于获取视觉时间表示的代理练习有关。

数据生成任务。生成对抗网络 中的表示能力也可以用于预训练任务。如 BiGANs [48] 所示，将数据投影回潜在空间有助于通过充当特征表示来辅助监督判别任务。

数据重构任务。由于受自然语言启发，图像可以分为补丁，因此一些用于 NLP 的预训练任务也可以用于 CV，例如基于自编码器的掩码预测。原始图像首先被分成几个补丁，并使用离散视觉标记对每个补丁进行编码。在第二阶段输出掩码补丁的视觉标记，以匹配来自固定分词器的相应视觉标记。

杂项。为了在 CV 中训练 PFMs，建议了其他预训练任务。例如，

基于对比学习，编码器网络用于在各种数据增强上进行预训练。通过最大化负对（例如，具有不同标签的对）之间的距离并最小化正对（例如，具有相同标签的对）之间的距离来训练参数。为了预训练主干网络的参数，DeepClustering [49] 方法将表示划分为不同的簇，并将这些簇标记为监督信号。

# 2.3.3 用于 GL 的预训练任务

GL 中的预置任务与其他代理任务类似。但是，根据设计，它们可以是监督的或无监督的。根据 GL 中的预训练目的和潜在动机，这些任务可以分为以下几类：

图信息补全。该任务指的是首先屏蔽输入图中的一部分信息，然后根据剩余信息的分布分析来恢复被屏蔽的信息。类似的任务也存在于 CV 和 NLP 中，它们的目标分别是填充隐藏的像素或单词。

图属性预测。与直接对输入图的信息进行建模不同，该任务旨在通过挖掘输入图的潜在属性来提供各种自监督信号。具体来说，一方面，它考虑节点属性、局部子结构和连通性信息来提供预测回归任务；另一方面，它通过簇、结构密度和属性相似性等信息为节点分配伪标签以提供分类任务。

图一致性分析。该任务的目标是最大化图中具有相似语义信息的样本在图嵌入中的一致性，并最小化具有不相关语义信息的样本之间的一致性。在实际场景中，可以根据不同的模型训练策略将其分为上下文/自身/跨尺度的一致性分析。

杂项。与仅使用一个代理任务相比，一些方法设计了一些集成机制，将多个代理任务的优势整合到一个统一的框架中。此外，特定领域的一些图数据具有实际意义的独特自监督信号，可用于针对性设计下的预训练。

总之，Transformer 是大模型架构的重要组成部分，它有助于学习重要特征并挖掘数据中的内在结构。可以根据数据集和特定任务使用不同的学习机制来训练 PFMs。特别是，考虑到各个领域大规模未标记数据的存在，SSL 是一种从数据中学习知识嵌入的有前途的机制。RL 通过根据奖励模型优化策略（模型）为下游任务微调 PFMs 提供了一种新方式。如何设计有效且高效的 PFM 任务以掌握数据背后的知识是一个重要的研究课题。

# 3 用于自然语言处理的PFM

NLP 是一个结合了语言学和计算机科学的研究领域。其主要研究任务包括词性标注、命名实体识别、语义角色标注、机器翻译、问答系统、情感分析、文本摘要、文本分类、关系抽取、事件抽取等。PFM 的思想首先在 NLP 中流行起来。随后，CV 和 GL 采用了这种有前景的预训练技术。PFM 在大规模基准数据集上进行训练，并在主要任务数据集上进行微调，以获得能够解决新的相似任务的模型。它同时对单词的句法和语义表示进行建模，并根据不同的输入上下文动态地改变多义词的表示。PFM 学习了丰富的语法和语义推理知识，并取得了更好的结果。在过去几年中，提出了许多 PFM，如表 1 所示。

在本节中，我们首先介绍单词表示学习模型，包括自回归语言模型（LM）、上下文 LM 和排列 LM。然后，我们介绍用于 PFM 设计方法和掩码设计方法的神经网络架构。此外，我们总结了用于提升模型性能、多任务学习和不同下游任务的增强方法。最后，我们介绍了指令对齐方法，例如 RLHF 和思维链，这些方法应用于 ChatGPT 等 PFM 中，以提供更符合人类偏好且危害性更小的输出。

# 3.1 单词表示方法

许多大规模预训练模型在问答、机器阅读理解和自然语言推理方面的表现已经超越了人类，这表明当前 PFM 的构建方法是行之有效的。现有的预训练 LM 主要根据单词表示方法分为三个分支：（1）自回归 LM，（2）上下文 LM，和（3）排列 LM。单词预测方向和上下文信息是这三个分支中最重要的因素。

**自回归语言模型** 自回归 LM 基于前面的单词预测下一个可能的单词，或者基于后面的单词预测最后一个可能的单词。它被选为特征提取器，文本表示从前面的单词中提取。因此，它在文本摘要和机器翻译等 NLG 任务中表现更好。对于一个序列 $T = [ w _ { 1 } , w _ { 2 } , \dots , w _ { N } ]$，给定单词的概率计算如下：

$$
p \left(w _ {1}, w _ {2}, \dots , w _ {N}\right) = \prod_ {i = 1} ^ {N} p \left(w _ {i} \mid w _ {1}, w _ {2}, \dots , w _ {i - 1}\right), \tag {7}
$$

其中 $i > 1$ 且 $N$ 是输入序列的长度。

GPT [50] 采用了自监督预训练和监督微调的两阶段方法，并使用堆叠的 Transformer [38] 作为其解码器。作为后续，OpenAI 团队继续扩展 GPT，提出了 GPT-2 [51]，并将堆叠的 Transformer 层数增加到 48 层。参数总数达到 15 亿。GPT-2 还引入了多任务学习 [52]。GPT-2 具有相当大的模型容量，可以针对不同的任务模型进行调整，而不是对其进行微调。然而，GPT-2 也使用自回归 LM。因此，它在没有急剧增加成本的情况下提高了模型性能。由于单向 Transformer 缺乏上下文建模能力，GPT-2 的主要性能提升来自于多任务预训练、超大数据集和超大模型的综合效应。特定的下游任务仍然需要基于任务的微调数据集。增加 LM 的训练规模可以显著提高与任务无关的性能。因此，GPT-3 [20] 被开发出来，其特点是模型规模为 1750 亿个参数，并使用 45 TB 的数据进行训练。这使其能够在无需针对特定下游任务进行微调的情况下表现出良好的性能。

**上下文语言模型** 自回归 LM 仅使用上方或下方的信息，不能同时使用上方和下方的信息。ELMO [53] 仅使用双向长短期记忆网络（LSTM），它是前向和后向两个单向 LSTM 的拼接。上下文 LM 的预测基于上下文单词。它使用 Transformer 编码器，由于自注意力机制，模型的上下层都彼此直接相连。对于单词序列 $T$，给定单词的概率计算如下：

$$
p \left(w _ {1}, w _ {2}, \dots , w _ {N}\right) = \prod_ {i = 1} ^ {N} p \left(w _ {i} \mid w _ {1}, w _ {2}, \dots , w _ {N}\right). \tag {8}
$$

BERT [13] 使用堆叠的多层双向 Transformer 作为基本结构，并使用 Word-Piece [54] 作为分词方法。模型输入由三部分组成：单词嵌入、分段嵌入和位置嵌入。它使用双向 Transformer 作为特征提取器，这弥补了 ELMO 和 GPT 的缺陷。然而，BERT 的缺点也不容忽视。双向 Transformer 结构并没有消除自编码模型的约束。其巨大的模型参数数量对于计算资源较低的设备非常不友好，并且难以部署和应用。此外，预训练中的掩码语言建模会导致与微调阶段模型输入的不一致性。大多数 PFM 需要更多的训练任务和更大的语料库。针对训练不足的问题，Liu 等人 [55] 提出了 RoBERTa。它使用了更大的批量大小和无标签数据。此外，它训练模型的时间更长，移除了 NSP 任务，并增加了长序列训练。在处理文本输入时，与 BERT 不同，它采用字节对编码（BPE）[56] 进行分词。BPE 对每个输入序列使用不同的掩码模式，即使输入序列是相同的。

**排列语言模型** 使用上下文 LM 的建模方法可以看作是自编码模型。然而，由于训练阶段和微调阶段的不一致性，自编码模型在自然语言生成（NLG）任务中的表现较差。排列 LM 旨在结合自回归 LM 和自编码器 LM 的优势。它在很大程度上改善了这两个模型的缺陷，并可以作为未来构建预训练目标任务的基本思路。对于给定的输入序列 $T = [ w _ { 1 } , w _ { 2 } . . . , w _ { N } ]$，排列 LM 目标函数的形式表示如下：

$$
\max  _ {\theta} \mathbb {E} _ {z \sim Z _ {N}} \left[ \sum_ {t = 1} ^ {N} \log p _ {\theta} \left(x _ {z _ {T = t}} \mid x _ {z _ {T <   t}}\right) \right], \tag {9}
$$

其中 $\theta$ 是所有排列中的共享参数，$Z _ { N }$ 表示输入序列 $T$ 的所有可能排列的集合，$z _ { T = t }$ 和 $z _ { T < t }$ 分别表示排列 $z \in Z _ { N }$ 的第 $t$ 个元素和 $[ 1 , 2 , \ldots , t - 1 ]$ 个元素。

以 BERT 为代表的 MLM 可以很好地实现双向编码。然而，MLM 在预训练期间使用掩码标记，但在微调期间不使用，这导致预训练和微调期间的数据不一致。为了实现双向编码并避免 MLM 的问题，提出了排列 LM。排列 LM 基于自回归 LM，避免了不一致数据的影响。然而，与传统的自回归模型不同，排列 LM 不再按顺序对序列进行建模。它给出序列的所有可能排列，以最大化序列的期望对数似然。通过这种方式，任何位置都可以利用来自所有位置的上下文信息，从而使排列 LM 实现双向编码。最常见的排列 LM 模型是 XLNET [14] 和 MPNet [57]。XLNET 是一种基于排列语言建模方法的 PFM，它融合了 Transformer-XL 的两项关键技术：相对位置编码和段循环机制。相比之下，MPNet 结合了掩码语言建模（MLM）和排列

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-12/925beae9-6fc7-404b-a575-971981ce47ad/9a3708b7ebc366642c3b34a410786e823e0a9337c784842e4a2cb173947df489.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-12/925beae9-6fc7-404b-a575-971981ce47ad/153da861831552a82c191ec57bf2d194dfd21ea95cddd05e35611149db1541b2.jpg)



图 3：BART [45] 的架构：推广了 BERT（由于双向编码器）、GPT（具有从左到右的解码器）。在使用双向模型对损坏的文档（左侧）进行编码后，使用自回归解码器来确定原始文档的可能性。


语言建模来预测 Token 依赖关系，使用辅助位置信息作为输入，使模型能够查看完整的句子并减少位置差异。这两个模型代表了 PFM 领域的重大进步。

# 3.2 模型架构设计方法

ELMO 采用多层 RNN 结构。每一层都是由前向和后向 LM 组成的双向 LSTM 结构。这两个方向的最大似然作为目标函数。与词向量方法相比，ELMO 引入了上下文信息并改善了多义词问题，但 ELMO 提取语言特征的整体能力较弱。

PFM 的应用研究主要有两个方向。一个是带微调的 PFM（例如 BERT），另一个是带零样本/少样本提示的 PFM（例如 GPT）。BERT 使用 Transformer 中的双向编码器来预测哪些单词被掩码，并确定两个句子是否构成上下文。然而，文档是双向编码的，缺失的 Token 是独立预测的，这降低了生成能力 [45]。GPT 使用自回归解码器作为特征提取器，基于前几个单词预测下一个单词，并使用微调解决下游任务，因此它更适合文本生成任务。然而，GPT 仅使用前面的单词进行预测，无法学习双向交互信息。

与这些模型不同，BART [45] 是一个由采用编码器-解码器结构的 seq2seq 模型构建的降噪自编码器，如图 3 [45] 所示。预训练主要包括使用噪声破坏文本和使用 seq2seq 模型重建原始文本。编码层采用双向 Transformer。它采用五种添加噪声的模式：（1）单词掩码；（2）单词删除；（3）跨度掩码；（4）句子重排；（5）文档重排。在编码器部分，序列在输入编码器之前已经被掩码。然后，解码器根据编码器输出的编码表示和未被掩码的序列恢复原始序列。一系列噪声模式的添加使得 BART 在序列生成和自然语言推理任务中的性能显著提高。

# 3.3 掩码设计方法

注意力机制首先将重要的单词聚合成句子向量，将重要的句子向量聚合成文本向量，这使得模型能够对不同的输入给予不同的关注 [58]。对于作为双向编码 LM 的 BERT，输入句子中的任何两个单词都可以看到彼此。然而，这阻碍了 BERT 模型学习 NLG 任务的能力。

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-12/925beae9-6fc7-404b-a575-971981ce47ad/158da3f8c19a79e660b658c82d362990dc482fb90c0fc804bbe31236aaf7b76e.jpg)



图 4：SpanBERT [44] 的架构。


Joshi 等人 [44] 基于 RoBERTa 提出了 SpanBERT，它采用了动态掩码和单段预训练的思想，如图 4 [44] 所示。还提出了跨度掩码和跨度边界目标（SBO）来掩码一定长度的单词。跨度边界目标任务是通过两端的观测 Token 恢复所有被掩码的跨度。训练阶段使用 RoBERTa 中提出的动态掩码策略，而不是在数据预处理期间进行掩码。与 BERT 不同，SpanBERT 随机覆盖连续的文本并添加 SBO 训练目标。它使用最接近跨度边界的 Token 来预测跨度，并消除了 NSP 预训练任务。

BERT 和 GPT 在 NLG 任务中只能分离训练编码器和解码器，而不能进行联合训练。Song 等人 [59] 提出了掩码 seq2seq 预训练模型 MASS。在训练阶段，编码器的输入序列被随机掩码为长度为 $k$ 的连续段。掩码段将通过 MASS 解码器恢复。UniLM [60] 通过为输入数据中的两个句子设计不同的掩码来完成 NLG 模型的学习。对于第一个句子，UniLM 使用与 Transformer 编码器相同的结构，使每个单词都能注意到其前面的单词和后面的单词。对于第二个句子，每个单词只能注意到第一个句子中的所有单词和当前句子中的前面单词。因此，模型输入的第一个和第二个句子形成了经典的 seq2seq 模式。

# 3.4 增强方法

**提升模型性能** 大多数流行的预训练模型需要大量的预训练数据，这对硬件提出了巨大的要求，使得重新训练具有挑战性，只能对模型进行微调。为了解决这些问题，出现了一些模型。例如，百度发布的 ERNIE Tiny 是一个小型化的 ERNIE [61]，它减少了层数，并将预测速度提高了 4.3 倍，而精度仅略有下降。Lan 等人提出了 ALBERT [47] 以减少内存消耗并提高训练速度。然而，不可否认的是，无论对这些大规模模型进行何种压缩，这些任务中的模型性能都会急剧下降。未来的工作需要注意高层语义和语法信息的高效表示以及无损压缩。通过使用词嵌入参数分解和层间隐藏参数共享，ALBERT 在没有性能损失的情况下显著减少了模型参数的数量。它提出了 SOP 训练任务，预测两个句子的顺序以提高性能。

**多任务学习的增强** ERNIE(Baidu) [61] 主要由两部分组成：Transformer 编码器和任务嵌入。在 Transformer 编码器中，使用自注意力机制来捕获每个 Token 的上下文信息并生成上下文表示嵌入。任务嵌入是一种将不同特征应用于任务的技术。ERNIE 2.0 [62] 引入多任务学习来实现词汇、语法和语义的预训练。ERNIE 2.0 使用七个不同的预训练任务，涵盖三个方面：词级、句级和语义级。它使用持续学习，使先前训练任务中的知识得以保留，并使模型能够获得长距离记忆。它使用 Transformer 编码器并引入任务嵌入，使模型能够在持续学习过程中区分不同的任务。UniLM [60] 使用三个预训练任务：单向 LM、双向 LM 和编码器-解码器 LM。它可以通过自注意力层掩码机制在预训练阶段同时完成三种目标任务。在训练阶段，UniLM 采用 SpanBERT 提出的小段掩码策略，损失函数由上述三个预训练任务的损失函数组成。为了保持在所有损失函数上的贡献一致性，三个预训练任务同时训练。多个任务的建模和参数共享使 LM 在自然语言理解（NLU）和 NLG 任务中获得了良好的泛化能力。

**针对不同下游任务的增强** 预训练模型往往规模较大，因此如何匹配不同的下游任务同样重要。出现了一些在专用语料库上训练的预训练模型 [63, 64, 65]。Cui 等人 [63] 提出了 BERT-whole word masking 模型（BERT-WWM）。他们直接在中文中使用 BERT，根据原始 MLM 训练进行随机掩码，导致语义信息丢失。由于中文中没有明确的语言边界，很容易丢失重要含义。ZEN [64] 是一个基于 BERT 的文本编码器，它采用 N-gram 来增强性能，有效地整合了相当粒度的文本信息，具有快速收敛速度和良好的性能。Tsai 等人 [65] 提出了一种面向多语言序列标注的模型，用于序列标注任务。采用知识蒸馏方法在两个任务中取得了更好的性能：针对多种低资源语言的词性标注和形态属性预测。推理时间缩短了 27 倍。

**示例：ChatGPT 和 Bard** 如图 5 所示，ChatGPT 使用 RLHF 基于 PFM GPT-3.5 进行微调。ChatGPT 使用与 InstructGPT 不同的数据收集设置。首先，收集一个包含提示词和期望输出行为的大型数据集。该数据集用于通过监督学习微调 GPT-3.5。其次，给定微调后的模型和一个提示词，模型将生成多个模型输出。标注者给出期望的分数并对输出进行排名以组成比较数据集，该数据集用于训练奖励模型。最后，使用近端策略优化（PPO）[66] RL 算法针对奖励模型优化微调后的模型。

另一个实验性对话 PFM，Bard 2，是由 Google 开发的。Bard 基于对话应用的 LM（LaMDA）。LaMDA [67] 构建在 Transformer 之上，在 1.56T 单词的对话数据和网络文本上进行了预训练。安全性和事实依据是对话 AI 的两个主要挑战，LaMDA 应用了使用高质量注释数据和外部知识源进行微调的方法来提高模型性能。

# 3.5 指令对齐方法

指令对齐方法旨在让 LM 遵循人类意图并生成有意义的输出。通用方法是以监督方式使用高质量语料库微调预训练 LM。为了进一步提高 LM 的有用性和无害性，一些工作将 RL 引入微调过程，以便 LM 能够根据人类或 AI 的反馈修改其响应。监督和 RL 方法都可以利用思维链 [24] 风格的推理来提高人类判断的性能和 AI 决策的透明度。

**监督微调（SFT）** SFT 是一种成熟的技术，用于释放知识并将其应用于特定的现实世界甚至未见过的任务。SFT 的模板由输入-输出对和指令 [113] 组成。例如，给定指令“Translate this sentence to Spanish:”和输入“The new office building was built in less than three months.”，我们希望 LM 生成目标“El nuevo edificio de oficinas se construyó en tres meses.”。模板通常是人工制作的，包括非自然指令 [114] 和自然指令 [115, 116]，或者基于种子语料库的自举 [117]。 LM 造成伤害的伦理和社会风险是 SFT 中的重大担忧 [118]。LaMDA 是迄今为止最大的 LM，因此依赖于众包工作者注释的数据，以便在三个对话类别中提供任何生成的 LaMDA 响应的安全评估：自然、敏感和对抗。规则列表用于进一步的安全微调和评估目的。

**来自反馈的强化学习** RL 已被应用于增强 NLP 任务中的各种模型，如机器翻译 [119]、摘要 [18]、对话生成 [120]、图像描述 [121]、问题生成 [122]、文本游戏 [123] 以及更多 [124, 125, 126]。RL 是一种通过将语言生成任务中的不可微分目标视为序列决策问题来对其进行优化的有效方法。然而，存在过度拟合使用神经网络的指标的风险，导致在指标上得分很高但毫无意义的样本 [127]。RL 也用于使 LM 与人类偏好对齐 [128, 129, 130]。

InstructGPT 提出使用 PPO 针对训练好的奖励模型对大模型进行微调，以使 LM 与人类偏好对齐 [19]，这与 ChatGPT 使用的方法相同，称为 RLHF。具体而言，奖励模型使用人工标注者对输出进行手动排名的比较数据进行训练。对于每一个输出，奖励模型或机器标注者计算一个奖励，用于使用 PPO 更新 LM。更多细节如图 5 所示。

PFM 技术最近的突破之一是 GPT-4 [25]，它遵循预训练方法来预测文档中的后续 Token，然后进行 RLHF 微调。随着任务复杂性的增加，GPT-4 在可靠性、创造力和处理更细致指令的能力方面优于 GPT-3.5。

由 DeepMind 开发的 Sparrow [130] 也利用了 RLHF，减少了不安全和不当答案的风险。尽管通过结合流畅性，使用 RLHF 取得了一些有希望的结果，但由于缺乏公开可用的基准测试和实现资源，该领域的进展受到阻碍，导致人们认为 RL 是 NLP 的一种困难方法。因此，最近引入了一个名为 RL4LMs [127] 的开源库，它包含用于在基于 LM 的生成上微调和评估 RL 算法的构建块。


表 1：NLP 中 PFM 的摘要。预训练任务包括语言模型（LM）、掩码 LM（MLM）、排列 LM（PLM）、降噪自编码器（DAE）、知识图谱（KG）和知识嵌入（KE）。


<table><tr><td>Year</td><td>Conference</td><td>Model</td><td>Architecture</td><td>Embedding</td><td>Training method</td><td>Code</td></tr><tr><td>2013</td><td>NeurIPS</td><td>Skip-Gram [68]</td><td>Word2Vec</td><td>Probabilistic</td><td>-</td><td>https://github.com/../models</td></tr><tr><td>2014</td><td>EMNLP</td><td>GloVe [69]</td><td>Word2Vec</td><td>Probabilistic</td><td>-</td><td>-</td></tr><tr><td>2015</td><td>NeurIPS</td><td>LM-LSTM [70]</td><td>LSTM</td><td>Probabilistic</td><td>LM</td><td>https://github.com/../GloVe</td></tr><tr><td>2016</td><td>IJCAI</td><td>Shared LSTM [71]</td><td>LSTM</td><td>Probabilistic</td><td>LM</td><td>https://github.com/../adversarial_text</td></tr><tr><td>2017</td><td>TACL</td><td>FastText [72]</td><td>Word2Vec</td><td>Probabilistic</td><td>-</td><td>https://github.com/../fastText</td></tr><tr><td>2017</td><td>NeurIPS</td><td>CoVe [73]</td><td>LSTM+Seq2Seq</td><td>Probabilistic</td><td>-</td><td>https://github.com/../cove</td></tr><tr><td>2018</td><td>NAACL-HLT</td><td>ELMO [53]</td><td>LSTM</td><td>Contextual</td><td>LM</td><td>https://allenlp.org/elmio</td></tr><tr><td>2018</td><td>NAACL-HLT</td><td>BERT [13]</td><td>Transformer Encoder</td><td>Contextual</td><td>MLM</td><td>https://github.com/../bert</td></tr><tr><td>2018</td><td></td><td>OpenAI GPT [50]</td><td>Transformer Decoder</td><td>Autoregressive</td><td>LM</td><td>https://github.com/../transformer-lm</td></tr><tr><td>2019</td><td>ACL</td><td>ERNIE(THU)</td><td>Transformer Encoder</td><td>Contextual</td><td>MLM</td><td>https://github.com/../ERNIE</td></tr><tr><td>2019</td><td>ACL</td><td>Transformer-XL [74]</td><td>Transformer-XL</td><td>Contextual</td><td>-</td><td>https://github.com/../transformer-xl</td></tr><tr><td>2019</td><td>ICLR</td><td>InfoWord [75]</td><td>Transformer Encoder</td><td>Contextual</td><td>MLM</td><td>-</td></tr><tr><td>2019</td><td>ICLR</td><td>StructBERT [76]</td><td>Transformer Encoder</td><td>Contextual</td><td>MLM</td><td>-</td></tr><tr><td>2019</td><td>ICLR</td><td>ALBERT [47]</td><td>Transformer Encoder</td><td>Contextual</td><td>MLM</td><td>https://github.com/../ALBERT</td></tr><tr><td>2019</td><td>ICLR</td><td>WKLM [77]</td><td>Transformer Encoder</td><td>Contextual</td><td>MLM</td><td>-</td></tr><tr><td>2019</td><td>ICML</td><td>MASS [59]</td><td>Transformer</td><td>Contextual</td><td>MLM(Seq2Seq)</td><td>https://github.com/../MASS</td></tr><tr><td>2019</td><td>EMNLP-IJCNLP</td><td>KnowBERT [78]</td><td>Transformer Encoder</td><td>Contextual</td><td>MLM</td><td>https://github.com/../kb</td></tr><tr><td>2019</td><td>EMNLP-IJCNLP</td><td>Unicoder [79]</td><td>Transformer Encoder</td><td>Contextual</td><td>MLM+TLM</td><td>-</td></tr><tr><td>2019</td><td>EMNLP-IJCNLP</td><td>MultiFit [80]</td><td>QRNN</td><td>Probabilistic</td><td>LM</td><td>https://github.com/../multifit</td></tr><tr><td>2019</td><td>EMNLP-IJCNLP</td><td>SciBERT [81]</td><td>Transformer Encoder</td><td>Contextual</td><td>MLM</td><td>https://github.com/../scibert</td></tr><tr><td>2019</td><td>EMNLP-IJCNLP</td><td>BERT-PKD [82]</td><td>Transformer Encoder</td><td>Contextual</td><td>MLM</td><td>https://github.com/../Compression</td></tr><tr><td>2019</td><td>NeurIPS</td><td>Xlnet [14]</td><td>Transformer-XL Encoder</td><td>Permutation</td><td>PLM</td><td>https://github.com/../xlnet</td></tr><tr><td>2019</td><td>NeurIPS</td><td>UNILM [60]</td><td>LSTM + Transformer</td><td>Contextual</td><td>LM + MLM</td><td>https://github.com/../unilm</td></tr><tr><td>2019</td><td>NeurIPS</td><td>XLM [83]</td><td>Transformer Encoder</td><td>Contextual</td><td>MLM+CLM+TLM</td><td>https://github.com/../XLM</td></tr><tr><td>2019</td><td>OpenAI Blog</td><td>GPT-2 [51]</td><td>Transformer Decoder</td><td>Autoregressive</td><td>LM</td><td>https://github.com/../gpt-2</td></tr><tr><td>2019</td><td>arXiv</td><td>RoBERTa [55]</td><td>Transformer Encoder</td><td>Contextual</td><td>MLM</td><td>https://github.com/../fairseq</td></tr><tr><td>2019</td><td>arXiv</td><td>ERNIE(Baidu) [61]</td><td>Transformer Encoder</td><td>Contextual</td><td>MLM+DLM</td><td>https://github.com/../ERNIE</td></tr><tr><td>2019</td><td>EMC2@NeurIPS</td><td>QBERTB [84]</td><td>Transformer Encoder</td><td>Contextual</td><td>MLM</td><td>https://github.com/../quantized_bert.py</td></tr><tr><td>2019</td><td>arXiv</td><td>DistillBERT [85]</td><td>Transformer Encoder</td><td>Contextual</td><td>MLM</td><td>https://github.com/../distillation</td></tr><tr><td>2020</td><td>ACL</td><td>fastBERT [86]</td><td>Transformer Encoder</td><td>Contextual</td><td>MLM</td><td>https://github.com/../FastBERT</td></tr><tr><td>2020</td><td>ACL</td><td>SpanBERT [44]</td><td>Transformer Encoder</td><td>Contextual</td><td>MLM</td><td>https://github.com/../SpanBERT</td></tr><tr><td>2020</td><td>ACL</td><td>BART [45]</td><td>Transformer</td><td>En: Contextual De: Autoregressive</td><td>DAE</td><td>https://github.com/../transformers</td></tr><tr><td>2020</td><td>ACL</td><td>CamemBERT [87]</td><td>Transformer Encoder</td><td>Contextual</td><td>MLM(WWM)</td><td>https://camembert-model.fr</td></tr><tr><td>2020</td><td>ACL</td><td>XLM-R [88]</td><td>Transformer Encoder</td><td>Contextual</td><td>MLM</td><td>https://github.com/../XLM</td></tr><tr><td>2020</td><td>ICLR</td><td>Reformer [89]</td><td>Reformer</td><td>Permutation</td><td>-</td><td>https://github.com/../reformer</td></tr><tr><td>2020</td><td>ICLR</td><td>ELECTRA [46]</td><td>Transformer Encoder</td><td>Contextual</td><td>MLM</td><td>https://github.com/../electra</td></tr><tr><td>2020</td><td>AAAI</td><td>Q-BERT [90]</td><td>Transformer Encoder</td><td>Contextual</td><td>MLM</td><td>-</td></tr><tr><td>2020</td><td>AAAI</td><td>XNLG [91]</td><td>Transformer</td><td>Contextual</td><td>MLM+DAE</td><td>https://github.com/../xnlg</td></tr><tr><td>2020</td><td>AAAI</td><td>K-BERT [92]</td><td>Transformer Encoder</td><td>Contextual</td><td>MLM</td><td>https://github.com/../K-BERT</td></tr><tr><td>2020</td><td>AAAI</td><td>ERNIE 2.0 [62]</td><td>Transformer Encoder</td><td>Contextual</td><td>MLM</td><td>https://github.com/../ERNIE</td></tr><tr><td>2020</td><td>NeurIPS</td><td>GPT-3 [20]</td><td>Transformer Decoder</td><td>Autoregressive</td><td>LM</td><td>https://github.com/../gpt-3</td></tr><tr><td>2020</td><td>NeurIPS</td><td>MPNet [57]</td><td>Transformer Encoder</td><td>Permutation</td><td>MLM+PLM</td><td>https://github.com/../MPNet</td></tr><tr><td>2020</td><td>NeurIPS</td><td>ConvBERT [93]</td><td>Mixed Attention</td><td>Contextual</td><td>-</td><td>https://github.com/../ConvBert</td></tr><tr><td>2020</td><td>NeurIPS</td><td>MiniLM [94]</td><td>Transformer Encoder</td><td>Contextual</td><td>MLM</td><td>https://github.com/../minlm</td></tr><tr><td>2020</td><td>TACL</td><td>mBART [95]</td><td>Transformer</td><td>Contextual</td><td>DAE</td><td>https://github.com/../mbart</td></tr><tr><td>2020</td><td>COLING</td><td>CoLAKE [96]</td><td>Transformer Encoder</td><td>Contextual</td><td>MLM+KE</td><td>https://github.com/../CoLAKE</td></tr><tr><td>2020</td><td>LREC</td><td>FlauBERT [97]</td><td>Transformer Encoder</td><td>Contextual</td><td>MLM</td><td>https://github.com/../Flaubert</td></tr><tr><td>2020</td><td>EMNLP</td><td>GLM [98]</td><td>Transformer Encoder</td><td>Contextual</td><td>MLM+KG</td><td>https://github.com/../GLM</td></tr><tr><td>2020</td><td>EMNLP (Findings)</td><td>TinyBERT [99]</td><td>Transformer</td><td>Contextual</td><td>MLM</td><td>https://github.com/../TinyBERT</td></tr><tr><td>2020</td><td>EMNLP (Findings)</td><td>RobBERT [100]</td><td>Transformer Encoder</td><td>Contextual</td><td>MLM</td><td>https://github.com/../RobBERT</td></tr><tr><td>2020</td><td>EMNLP (Findings)</td><td>ZEN [64]</td><td>Transformer Encoder</td><td>Contextual</td><td>MLM</td><td>https://github.com/../ZEN</td></tr><tr><td>2020</td><td>EMNLP (Findings)</td><td>BERT-MK [101]</td><td>KG-Transformer Encoder</td><td>Contextual</td><td>MLM</td><td>-</td></tr><tr><td>2020</td><td>RepL4NLP@ACL</td><td>CompressingBERT [35]</td><td>Transformer Encoder</td><td>Contextual</td><td>MLM(Pruning)</td><td>https://github.com/../bert-prune</td></tr><tr><td>2020</td><td>JMLR</td><td>T5 [102]</td><td>Transformer</td><td>Contextual</td><td>MLM(Seq2Seq)</td><td>https://github.com/../transformer</td></tr><tr><td>2021</td><td>T-ASL</td><td>BERT-wwm-Chinese [63]</td><td>Transformer Encoder</td><td>Contextual</td><td>MLM</td><td>https://github.com/../BERT-wwm</td></tr><tr><td>2021</td><td>EACL</td><td>PET [103]</td><td>Transformer Encoder</td><td>Contextual</td><td>MLM</td><td>https://github.com/../pet</td></tr><tr><td>2021</td><td>TACL</td><td>KEPLER [104]</td><td>Transformer Encoder</td><td>Contextual</td><td>MLM+KE</td><td>https://github.com/../KEPLER</td></tr><tr><td>2021</td><td>EMNLP</td><td>SimCSE [105]</td><td>Transformer Encoder</td><td>Contextual</td><td>MLM+KE</td><td>https://github.com/../SimCSE</td></tr><tr><td>2021</td><td>ICML</td><td>GLaM [106]</td><td>Transformer</td><td>Autoregressive</td><td>LM</td><td>-</td></tr><tr><td>2021</td><td>arXiv</td><td>XLM-E [107]</td><td>Transformer</td><td>Contextual</td><td>MLM</td><td></td></tr><tr><td>2021</td><td>arXiv</td><td>T0 [108]</td><td>Transformer</td><td>Contextual</td><td>MLM</td><td>https://github.com/../T0</td></tr><tr><td>2021</td><td>arXiv</td><td>Gopher [109]</td><td>Transformer</td><td>Autoregressive</td><td>LM</td><td>-</td></tr><tr><td>2022</td><td>arXiv</td><td>MT-NLG [110]</td><td>Transformer</td><td>Contextual</td><td>MLM</td><td>-</td></tr><tr><td>2022</td><td>arXiv</td><td>LaMDA [67]</td><td>Transformer Decoder</td><td>Autoregressive</td><td>LM</td><td>https://github.com/../LaMDA</td></tr><tr><td>2022</td><td>arXiv</td><td>Chinchilla [111]</td><td>Transformer</td><td>Autoregressive</td><td>LM</td><td>-</td></tr><tr><td>2022</td><td>arXiv</td><td>PaLM [43]</td><td>Transformer</td><td>Autoregressive</td><td>LM</td><td>https://github.com/../PaLM</td></tr><tr><td>2022</td><td>arXiv</td><td>OPT [112]</td><td>Transformer Decoder</td><td>Autoregressive</td><td>LM</td><td>https://github.com/../MetaSeq</td></tr></table>

除了人类反馈之外，最新的对话代理之一——Claude 倾向于宪法 AI [131]，其中奖励模型通过来自 AI 反馈的 RL（RLAIF）学习。批评和 AI 反馈都由从“宪法”中提取的一小部分原则指导，这是在 Claude 中唯一由人类提供的东西。AI 反馈专注于通过解释其对危险查询的反对意见来控制输出以减少危害。

**思维链** 思维链（CoT）提示是一种通过提示大语言模型（LLM）生成导致多步骤问题最终答案的一系列中间步骤来提高其推理能力的技术。CoT 是一系列中间推理步骤，可以显著提高 LLM 执行复杂推理的能力 [24, 132, 133]。此外，与没有 CoT 相比，使用 CoT 进行微调显示出危害略小 [131]。CoT 提示是模型规模的涌现属性，意味着它对于更大型、更强大的语言模型效果更好。也可以在 CoT 推理数据集上微调模型以进一步增强此能力并激发更好的可解释性。

在 CoT 提示实验中，向模型提供一个提示，概述了一个多步骤问题。提示可能会提出一个问题，例如“卖掉了100只鸡中的30只和20只猪中的10只后，农夫还剩下多少只动物？”。然后模型生成一系列中间推理步骤，例如“农夫还剩下 $100 - 30 = 70$ 只鸡”和“农夫还剩下 $20 - 10 = 10$ 只猪”，然后生成最终答案，例如“农夫还剩下 $70 + 10 = 80$ 只动物”。CoT 提示已证明在提高 LLM 在各种推理任务（如算术、符号推理和常识）方面的性能是有效的。这是一种有前途的技术，可以增强语言模型推理复杂问题的能力。

# 3.6 总结

神经概率 LM 使用神经网络来估计概率 LM 的参数，这减少了模型参数的大小，同时增加了上下文窗口的数量。在神经网络的帮助下，LM 不需要通过改进平滑算法来持续缓解性能瓶颈。由于训练目标是无监督的，因此具有大量数据的语料库就足以进行训练。训练过程中的负采样技术为 LM 中目标任务的后续研究提供了新思路。此外，神经概率 LM 由于其良好的表示能力和训练效率，促进了下游任务研究的进一步发展。在预训练 LM，特别是 BERT 模型提出之后，语言建模的研究进入了一个新阶段。双向 LM 采用的双向 LM、掩码 LM 和排列 LM 在更深层次上成功建模了自然语言中的语法和语义信息。ChatGPT 是 PFM 中使用 RL 的另一个里程碑式工作。PFM 的展示能力在定性上优于神经概率 LM。它在某些任务上甚至超过了人类。

# 4 计算机视觉的 PFMs

随着 PFM 在 NLP 中的普及，这激励着研究人员开始在 CV 领域探索 PFM。术语“pretraining”在 CV 领域的深度学习研究中尚未被明确定义。该词最初用于基于卷积的网络，指在 ImageNet 等更通用的数据集上调整参数，这使得其他任务可以以热启动初始化开始训练，从而更快地收敛。与早期依赖带有监督信号的预训练数据集的基于 CNN 的迁移学习技术相比，我们对 PFM 的考察主要集中在利用人工设计标签的 SSL 上，

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-12/925beae9-6fc7-404b-a575-971981ce47ad/34900fad144dcc386080c709fa187d8f3a37b291a12463d421d2071eb3d37fd3.jpg)

图 6：SSL 的通用流程。顶部代表预训练，底部流从上方获取迁移参数以学习下游监督任务。

例如拼图游戏，或比较来自图像的不同补丁作为 pretext tasks。这使得学到的表征可以泛化到各种下游任务，包括分类、检测、识别、分割等。

然而，当学习任务变得更加复杂时，依赖数据注释的成本很高，使得标记过程比实际学习更加艰巨和耗时。这是 SSL 迫切需要的地方，也是它如何进一步推动深度学习方法进步的方式。为了减少对数据标记的依赖，在 SSL 中，未标记的数据通过匹配、对比或生成的方式进行自监督训练。

SSL 的通用流程如图 6 所示。在预训练阶段，为编码器网络设计了一个 pretext task 来解决。该 pretext task 的人工标签是根据数据的特定属性自动生成的，例如来自同一来源的图像补丁被标记为“正”，而来自不同来源的补丁被标记为“负”。然后，通过监督学习方法训练编码器网络来解决这个 pretext task。由于浅层提取边缘、角度和纹理等细粒度细节，而深层捕获语义信息或图像内容等与任务相关的高级特征，因此在 pretext tasks 上学到的编码器可以迁移到下游监督任务。在此阶段，主干网络的参数是固定的，只需要学习一个简单的分类器，例如双层多层感知机 (MLP)。考虑到下游训练阶段的工作量有限，该学习过程通常被称为微调。总之，SSL 在预训练阶段学到的表征可以在其他下游任务中重用并取得相当的结果。

在本节中，我们介绍了在 CV 中预训练 PFMs 的不同任务。PFMs 可以通过特定的 pretext tasks、帧顺序、生成、重建、记忆库、共享、聚类等方式进行训练。我们在表 2 中总结了在 CV 中提出的 PFMs。

# 4.1 通过特定 Pretext Task 学习

在无监督学习的早期，网络通过设计一个特殊的 pretext task 并预测该任务的答案来进行训练。Dosovitskiy 等人 [134, 135] 预训练了 Exemplar CNN 来从未标记数据中区分不同的补丁。实验证明这些设计可以学习到可迁移到标准识别任务的有用表征。在基于上下文预测的方法 [136] 中，关于位置信息的人工监督信号作为对分类的标签。Inpainting [137] 旨在通过预测缺失的中心部分来预训练模型。由于 Inpainting 是一种基于语义的预测，因此在这种方式下将另一个解码器链接到上下文编码器。此外，解码器的标准逐像素重建过程可以迁移到任何其他下游 Inpainting 任务。具体而言，Colorization [138] 是一种评估作为 pretext task 的着色如何帮助学习下游任务的语义表征的方法。它也被称为跨通道编码，因为不同的图像通道作为输入，而输出是被区分的。同样，Split-Brain Autoencoder [139] 也通过强制网络解决跨通道预测任务以自监督方式学习表征。Jigsaw [140] 被提出来通过首先将拼图游戏设计为 pretext task，以自监督方式预训练设计的无上下文网络 (CFN)。完成破损拼图游戏 (CDJP) [141] 通过进一步使 pretext tasks 复杂化来学习图像表征，其中拼图缺失一块，其他块包含不完整的颜色。遵循设计高效且有效的 pretext tasks 的想法，Noroozi 等人 [142] 使用计算视觉基元作为一种特殊的 pretext task，并在常规基准测试中超越了以前的 SOTA 模型。NAT [143] 通过将主干 CNN 的输出与低维噪声对齐来学习表征。RotNet [144] 旨在预测图像的不同旋转角度。

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-12/925beae9-6fc7-404b-a575-971981ce47ad/12ea1c0a832256e076eb29104151c54ecfc8d312e8e649d04c3c32d322746217.jpg)

图 7：对比预测编码 [145]。输入序列可以表示图像和视频。

# 4.2 通过帧顺序学习

诸如视频之类的序列数据的学习总是涉及通过时间步的帧处理。这个问题通常与解决有助于学习视觉时间表征的 pretext tasks 有关。对比预测编码 (CPC) [145] 是第一个通过在潜在空间中预测未来来学习数据表征的模型。该模型可以输入任何模态的数据，如语音、图像、文本等。CPC 的组件如图 7 [145] 所示，其中 $x _ { t }$ 表示观测的输入序列，$z _ { t }$ 是编码器 $g _ { e n c }$ 之后的潜在表征序列，$c _ { t }$ 是在自回归模型 $g _ { a r }$ 之后总结所有潜在序列 $z _ { \le t }$ 的上下文潜在表征。与传统模型通过生成模型 $p _ { k } \big ( x _ { t + k } \big | c _ { t } \big )$ 预测未来帧 $x _ { t + k }$ 不同，CPC 对“密度比”$f _ { k }$ 进行建模，以表示上下文潜在表征 $c _ { t }$ 和未来帧 $x _ { t + k }$ 之间的互信息：

$$
f _ {k} \left(x _ {t + k}, c _ {t}\right) \propto p \left(x _ {t + k} \mid c _ {t}\right) / x _ {t + k}. \tag {10}.
$$

在循环神经网络编码之后，$z _ { t }$ 和 $c _ { t }$ 都可以根据需要为下游任务进行选择。编码器和自回归模型通过 InfoNCE [145] 进行训练，如下所示

$$
\mathcal {L} = - \mathbb {E} _ {X} \left[ \log f _ {k} \left(x _ {t + k}, c _ {t}\right) / \sum_ {x _ {j} \in X} f _ {k} \left(x _ {j}, c _ {t}\right) \right], \tag {11}.
$$

其中 $X$ 表示包含正负样本的训练数据集。密度比 $f _ { k }$ 可以通过优化 $\mathcal { L }$ 来估计。CPC v2 通过在无监督表征上进行预训练来重新审视和改进 CPC [146]，其表征泛化能力可以迁移到数据高效的下游任务。

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-12/925beae9-6fc7-404b-a575-971981ce47ad/04ba6492011ce2c6fe5cb7eaf2b4683f5577cf91e2d9c3ed1ddfb4022154b353.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-12/925beae9-6fc7-404b-a575-971981ce47ad/94ae9969739fc9d0b92755e176b4d22df0bfa5e76758ea40ccef89a157996a22.jpg)

图 8：BigBiGAN 框架的结构 [147]。

# 4.3 通过生成学习

尽管 GAN 方法的发展使许多现有应用变得流行，但由于缺乏特征编码器，GAN 内部的表征能力并未被完全利用。因此，提出了双向生成对抗网络 [48] 将数据投影回潜在空间，这通过作为特征表征对辅助监督判别任务很有用。

基于 BiGANs，BigBiGAN [147] 通过添加编码器并修改判别器，首次在 ImageNet 的无监督表征学习中实现了 SOTA。如图 8 [147] 所示，GAN 的传统组件（编码器 $\mathcal { E }$ 和生成器 $\mathcal { G }$ ）用于生成数据-潜在对，表示为 $( \mathbf { x } \sim P _ { \mathbf { x } } , \hat { \mathbf { z } } \sim \mathcal { E } ( \mathbf { x } ) )$ 和 $( \hat { \mathbf { x } } \sim \mathcal { G } ( \mathbf { z } ) , \mathbf { z } \sim P _ { \mathbf { z } } )$。最终损失 $\ell$ 定义为数据特定项 $s _ { \mathbf { X } } , s _ { \mathbf { Z } }$ 和数据联合项 $s _ { \mathbf { x } \mathbf { z } }$ 之和。引入的判别器 $\mathcal { D }$（对抗学习推理 (ALI) [148] 或 BiGAN [48]）学习区分来自原始数据、潜在分布和编码向量的对。

# 4.4 通过重建学习

iGPT [149] 和 ViT [40] 模型证明了使用自编码器将遮蔽预测的 pretext task 从语言适应到图像数据的可行性。BEiT [150] 首次证明了基于自编码器的遮蔽预测可以优于 DINO [151]，后者是一种不使用预训练技术的传统 SOTA 方法。具体而言，BEiT 包括两个阶段：使用离散变分自编码器 (dVAE) [152] 进行 token 嵌入，以及使用遮蔽图像预测进行 tokenizer 训练。在第一阶段，原始图像被分成一些补丁并使用离散 token 进行编码，这与 BERT 不同，因为图像补丁不像 NLP 中的单词那样有现成的 token。在第二阶段，BEiT 编码器接收一个包含未遮蔽和遮蔽补丁的损坏图像，然后输出遮蔽补丁的视觉 token 以匹配来自固定 tokenizer 的相应视觉 token。尽管取得了成功，但遮蔽预测和自编码器训练之间的分离导致整个框架不是端到端的，并阻碍了学习的有效性和效率。

为了解决这个问题，MAE [154] 提出了一种端到端的简单解决方案，通过均方误差 (MSE) 损失直接从未遮蔽补丁预测遮蔽补丁。值得注意的是，MAE 使用 $7 5 \%$ 的遮蔽率，这显着高于 BERT（通常为 $1 5 \%$）。消融研究表明，更高的遮蔽率对微调和线性探测都有益。与此同时，SimMIM [155] 提出了一种与 MAE 类似的基于自编码器的解决方案，其中他们也确认

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-12/925beae9-6fc7-404b-a575-971981ce47ad/058aba0532035ac6772ec976f42cb9a0bb1250891a7b1422fda8ba68af0f7eb1.jpg)

图 9：记忆库方法的通用流程 [153]。

更高的标记率和利用随机遮蔽策略有助于提高性能。主要区别在于他们如何划分自编码器中表征编码和 pretext prediction 的责任。由于 SimMIM 的解码器很简单，SimMIM 的编码器同步执行这两者。相反，MAE 中的编码器仅承担表征编码的角色，而解码器负责 pretext prediction。最近，Meta AI 宣布了 Segment Anything Model (SAM) [156]，它提示用户指定图像中要分割的内容，从而允许进行广泛的分割任务而无需额外训练。SAM 采用了一个 MAE 预训练的 ViT-H [40] 图像编码器，每张图像运行一次并产生图像嵌入，以及一个嵌入点击或框等输入提示的提示编码器。随后，一个基于轻量级 transformer 的掩码解码器从图像和提示嵌入中预测对象掩码。结果表明，SAM 可以从单个前景点生成高质量的掩码，这些掩码通常仅比手动标注的基准实况略逊一筹。它通常使用零样本迁移方法和提示工程在广泛的下游任务上实现强大的定量和定性结果。

在 MAE 中利用 ViT 带来了严重的低效问题，即减小补丁大小会导致计算资源的二次方增加。为了解决这个问题，有两个重要的解决方案：(1) 分层 ViT 和 (2) 局部注意力。在第一个方向，引入了分层 ViT (hViT)，它利用收缩的金字塔结构和移位窗口 [157] 等技术来减少计算需求。不幸的是，hViT 不能直接应用于实现 MAE 预训练，因为 hViT 中使用的局部窗口注意力使其难以像 MAE 那样处理随机遮蔽的补丁。最近，提出了 Uniform Masking MAE (UM-MAE) [158] 以使用 hViT 增强 MAE，它引入了两阶段流程：采样和遮蔽。它首先从每个块中随机采样一部分补丁（论文中报告为 $2 5 \%$），然后在采样的补丁之上遮蔽额外的补丁。第一步有助于在不同局部窗口之间保持公共元素，而第二步防止通过附近的低级特征进行像素重建的捷径，从而使任务更加困难。另一个提高效率的方向是通过将网络的注意力放在图像的一些局部小窗口中来减少输入大小。受局部知识足以重建遮蔽补丁的观察启发，提出了局部遮蔽重建 (LoMaR) [159]。LoMaR 不使用整幅图像进行掩码重建，而是采样多个小窗口并将注意力集中在局部区域，在学习效率方面在下游任务上优于 MAE。

# 4.5 通过记忆库学习

非参数实例判别 (NPID) [153] 是第一种利用实例学习下游任务表征的方法。详细流程如图 9 所示。特征表征存储在记忆库中以便于计算，因为实例级分类目标需要训练数据集中的所有图像。对于任何具有特征表征 $\mathbf { v } = f _ { \boldsymbol { \theta } } ( \boldsymbol { x } )$ 的图像 $x$，

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-12/925beae9-6fc7-404b-a575-971981ce47ad/b017e0e894b5e566435d68d5b2338b70de4eb1d394d4ea3d6e8b07d96e828359.jpg)

图 10：所有双流模型的总结，包括对比学习和基于记忆库的方法。

其被识别为第 $i$ 个示例的概率为：

$$
P (i | \mathbf {v}) = e x p \left(\mathbf {v} _ {i} ^ {\mathrm {T}} \mathbf {v} / \tau\right) / \sum_ {j = 1} ^ {n} e x p \left(\mathbf {v} _ {j} ^ {\mathrm {T}} \mathbf {v} / \tau\right), \tag {12}.
$$

其中 ${ \bf v } _ { i }$ 或 ${ \bf v } _ { j }$ 是第 $i$ 个或第 $j$ 个样本的表征，作为参数类原型（即分类器的权重）的替代品。此外，$\tau$ 是从知识蒸馏 [160] 借用的温度参数。

局部聚合 (LA) [161] 是另一种训练 CNN 编码器将原始图像嵌入到低维空间——嵌入空间的方法。当局部聚合的度量最大化时，相似的数据实例在嵌入空间中移动在一起，而不相似的实例分开移动。

基于 NPID，提出了 Pretext 不变表征学习 (PIRL，发音为“pearl”) [162]，以论证语义表征在 pretext 变换任务下是不变的。假设图像的原始视图和变换视图分别表示为 $I$ 和 $I ^ { t }$。这些样本视图被输入到 CNN 编码器，训练数据集 $\mathcal { D }$ 上的总经验损失可以定义为：

$$
\mathcal{L} _ {\text {t o t a l}} (\theta ; \mathcal{D}) = \mathbb {E} _ {t \sim \mathcal{T}} \left[ \frac {1}{| \mathcal{D} |} \sum_ {\boldsymbol {I} \in \mathcal{D}} \mathcal{L} \left(\boldsymbol{V} _ {\boldsymbol{I}}, \boldsymbol{V} _ {\boldsymbol{I} ^ {t}}\right) \right], \tag {13}.
$$

其中 $\tau$ 表示图像的不同变换。损失鼓励图像 I 的表征与 $I ^ { t }$ 的表征相似，而 $I ^ { t }$ 的表征与不同图像 $\pmb { I ^ { \prime } }$ 的表征不相似，如图 10 的虚线框所示。因此，更多的负样本对有助于提高梯度的可扩展性，并导致最终学到的编码器具有更强的表征能力。这就是引入记忆库以存储更多先前表征用于后续比较的原因。

# 4.6 通过共享学习

SSL 倾向于为不同的数据增强使用两个编码器网络，然后通过最大化负对之间的距离或最小化正对之间的距离来预训练参数。图 10 显示了所有对比学习框架的双流模型。对原始输入图像 $\pmb { I }$ 的变换 $t$ 生成视图 $v$，同样，其对应的 $t ^ { \prime }$ 生成 $v ^ { \prime }$。通常，使用两个不同或相同的编码器 $f _ { \theta }$ 和 $f _ { \xi } ^ { \prime }$ 来提取对比表征。随后的 MLP 头部 $g _ { \boldsymbol { \theta } }$ 和 $g _ { \xi } ^ { \prime }$ 用于学习更多有利于对比损失的组合。值得注意的是，MLP 和记忆库可以在不同设置下被移除或保留。就共享编码器而言，SSL 可以分为两类：1) 软共享，即两个编码器共享相似但不同的参数 $( f _ { \theta } \neq f _ { \xi } ^ { \prime } )$；2) 硬共享，即两个编码器保持相同的架构和参数 $( f _ { \theta } = f _ { \xi } ^ { \prime } )$。

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-12/925beae9-6fc7-404b-a575-971981ce47ad/9849e6aa5b73395e6ccc4655b13a090a3a41dcf0ac7d954eafe9e43c03e9f312.jpg)

图 11：MoCo [163] 的通用流程，它也是一个具有不同参数的双流框架。

软共享。Facebook AI Research (FAIR) 通过使用动量控制两个编码器之间的细微差异，提出了动量对比 (MoCo) [163]。如图 11 所示，其中一个编码器作为字典查找任务，生成编码数据样本队列 $\{ k _ { 0 } , k _ { 1 } , \cdot \cdot \cdot \}$。另一个编码器随着训练批次的更新生成编码查询 $\{ q _ { 0 } , q _ { 1 } , \cdot \cdot \cdot \}$。相似度通过新传入的编码查询 $q$ 与存储在字典队列中的编码键的点积来测量。假设在新键到来之前，队列中存储了 $K$ 个键。这 $K$ 个键被视为新键查询的负样本。为了结合负样本和正样本上的对比损失，在 MoCo 中使用 InfoNCE 损失 [145] 进行预训练。MoCo 中用于软参数共享的关键设计称为动量更新。He 等人 [163] 建议，键编码器（即动量编码器）到查询编码器的直接参数变化会失去必要的一致性并导致结果不佳。动量编码器参数 $\theta _ { k }$ 更新如下：

$$
\theta_ {k} = m \theta_ {k} + (1 - m) \theta_ {q}, \tag {14}.
$$

其中查询编码器参数 $\theta _ { q }$ 直接来自新传入实例的梯度学习，$m \in [ 0 , 1 )$ 是一个控制一致性的超参数（如果 $m$ 更接近 1，$\boldsymbol { \theta _ { k } }$ 更一致）。

受 SimCLR [164] 设计的启发，在 MoCo v2 [164] 中，FAIR 团队在编码器之后引入了 MLP 投影头，并利用了更多的数据增强技术来提高性能。进一步的改进来自：1) 嵌入式线性分类器缩小了无监督和监督预训练表征之间的差距；2) 更大的训练批次和更强的数据增强使更多的对比样本成为可能。

DeepMind 提出了 Bootstrap Your Own Latent (BYOL) [165]，它包含表征、投影和判别阶段，在不使用负样本的情况下实现了新的 SOTA。他们认为原始图像不同视图之间的判别是防止预训练期间崩溃的必要手段。然而，他们认为许多负样本对于防止这种崩溃并不是不可或缺的。如图 10 左侧所示，BYOL 中有两个具有不同参数的流。在线网络（顶部绿色）通过比较自身生成的预测与目标网络提供的回归目标来更新参数。然后目标模型（底部红色）的参数更新与公式 (14) 相同，即 $\xi  \tau \xi + ( 1 - \tau ) \theta$，其中 $\tau$ 是目标衰减率，用于控制目标网络中参数变化的程度。因此，目标网络也可以理解为动量编码器。这里，目标模型中的 $\xi$ 是动量编码器中的参数 $\theta _ { k }$，在线网络中的 $\theta$ 表示查询编码器中的参数 $\theta _ { q }$。

硬共享。SimCLR [166] 由 Google Research 的 Brain Team 提出，它利用了硬参数共享架构。这个简单的框架也可以在图 10 中总结，其中我们可以

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-12/925beae9-6fc7-404b-a575-971981ce47ad/100f239013124fd6b64d7daeac07d7fbec8940d7a7fb7027666c474c85cedfd6.jpg)

图 12：DeepCluster 模型的关键流程 [49]。

看到同一图像的不同视图的表征是在网络 $f ( \cdot )$ 中学习的。该基础编码器彼此共享参数。因此，不需要记忆库和动量设置来学习键和查询编码器，这有助于更简单的主干架构和更容易的学习策略。最大化同一图像的不同视图（正对）之间相似度的损失函数定义为

$$
\ell_ {i, j} = - \log e x p \left(\sin \left(z _ {i}, z _ {j}\right) / \tau\right) / \sum_ {k = 1} ^ {2 N} \mathbb {1} _ {[ k \neq i ]} e x p \left(\sin \left(z _ {i}, z _ {k}\right) / \tau\right), \tag {15}.
$$

其中 $( i , j )$ 是一对正样本，$\tau$ 是一个引入的超参数，称为温度参数 [153]，$\mathbb { 1 } _ { [ k \neq i ] } \in \{ 0 , 1 \}$ 是一个指示函数，用于控制分母仅包含负对。

为了避免对大量显式成对特征比较的依赖，Inria 和 FAIR 提出了同一图像的多个视图之间的交换分配 (SwAV) [167] 作为一种在线算法。SwAV 引入聚类来替代以前的对之间比较，这在非队列架构的帮助下获得了更多内存。在这种方法中，聚类原型参与定义的损失函数的计算。该原型被编码为通过 CNN 中的反向传播学习的向量的串联。因此，SwAV 不需要比较不同视图之间的编码表征。

基于现有的 SwAV，一种称为 SElf-supERvised (SEER) [168] 的新模型旨在从野外的任何随机图像和无界数据集中学习预训练编码器。基础网络是使用 SwAV SSL 方法 [167] 训练的 RegNetY 架构 [169]。该方法证明 SSL 并不特定于 ImageNet 等精选数据集，并且最近的 RegNet 的可扩展性释放了 ResNet 等传统主干网络的限制。此外，该方法鼓励研究界探索更多适合通用 SSL 的主干网络。

在最近的 SSL 中引起了注意，FAIR 利用简单孪生 (SimSiam) 网络的结构对 SSL 进行了实证实验。这种方法 [170] 可以避免传统对比学习中负样本对、大批次（或记忆库）和动量编码器的设计。图 10 中两个具有相同参数的编码器处理图像 $x$ 的两个不同视图 $t$ 和 $t ^ { \prime }$ 被唯一的孪生网络所取代。MLP 预测器 $g$ 用于其中一个视图表征，然后对另一个视图表征应用停止梯度操作。

# 4.7 通过聚类学习

DeepCluster [49] 是第一个采用聚类算法进行大规模数据集学习的模型。该方法将表征分组到不同的聚类中，并将这些聚类标记为监督信号以


表 2：CV 中 PFMs 的总结。

<table><tr><td>Year</td><td>Conference</td><td>Method</td><td>Pretext Task</td><td>Architecture</td><td>Downstream \(Task^1\)</td><td>Code</td></tr><tr><td>2014</td><td>NeurIPS</td><td>Exemplar-CNN [134, 135]</td><td>discrimination</td><td>CNN</td><td>cla, rec</td><td>https://lmb.informatik.uni-freiburg.de/...</td></tr><tr><td>2015</td><td>ICCV</td><td>Context [136]</td><td>context prediction</td><td>CNN</td><td>cla, det,CLU</td><td>https://github.com/../deepcontext</td></tr><tr><td>2016</td><td>CVPR</td><td>Inpainting [137]</td><td>inpainting</td><td>GAN, CNN</td><td>cla, det,seg,inp</td><td>https://github.com/../context-encoder</td></tr><tr><td>2016</td><td>ECCV</td><td>Colorization [138]</td><td>colorization</td><td>CNN</td><td>cla, det,seg</td><td>https://github.com/../colorization</td></tr><tr><td>2016</td><td>ECCV</td><td>Jigsaw [140]</td><td>Jigsaw puzzles</td><td>CNN</td><td>cla, det,seg,ret</td><td>https://github.com/../JigsawPuzzleSolver</td></tr><tr><td>2017</td><td>CVPR</td><td>Split-Brain [139]</td><td>channel prediction</td><td>CNN</td><td>cla, det,seg</td><td>https://richzhang.github.io/splittrainauto</td></tr><tr><td>2017</td><td>ICCV</td><td>Counting [142]</td><td>counting</td><td>CNN</td><td>cla, det,seg,ret</td><td>https://github.com/clvrai/...</td></tr><tr><td>2017</td><td>ICML</td><td>NAT [143]</td><td>noise</td><td>CNN</td><td>cla, det</td><td>-</td></tr><tr><td>2017</td><td>ICLR</td><td>BiGAN [48]</td><td>generation</td><td>GAN, CNN</td><td>cla, det,seg</td><td>https://github.com/../bigan</td></tr><tr><td>2018</td><td>WACV</td><td>CDJP [141]</td><td>Jigsaw puzzles</td><td>CNN</td><td>cla, det,seg</td><td>-</td></tr><tr><td>2018</td><td>ICLR</td><td>RotNet [138]</td><td>rotation</td><td>NIN, CNN</td><td>cla, det,seg</td><td>https://github.com/gidariss/...</td></tr><tr><td>2018</td><td>arXiv</td><td>CPC [145]</td><td>patch overlapping</td><td>CNN, GRU</td><td>cla</td><td>-</td></tr><tr><td>2018</td><td>CVPR</td><td>NPID [153]</td><td>instance discrimination</td><td>CNN</td><td>cla</td><td>https://github.com/../lemniscate.pytorch</td></tr><tr><td>2018</td><td>ECCV</td><td>DeepCluster [49]</td><td>clustering</td><td>CNN</td><td>cla, det,seg</td><td>https://github.com/../deepcluster</td></tr><tr><td>2019</td><td>ICCV</td><td>LA [161]</td><td>local aggregation</td><td>CNN</td><td>rec, det</td><td>https://github.com/../LocalAggregation</td></tr><tr><td>2019</td><td>NeurIPS</td><td>BigBiGAN [147]</td><td>generation</td><td>GAN, CNN</td><td>gen,cla</td><td>https://tfhub.dev/..bigbigan</td></tr><tr><td>2019</td><td>CVPR</td><td>AET [172]</td><td>transformation</td><td>CNN</td><td>cla</td><td>https://github.com/../AET</td></tr><tr><td>2019</td><td>NeurIPS</td><td>AMDIM [173]</td><td>discrimination</td><td>CNN</td><td>cla</td><td>https://github.com/../amdim-public</td></tr><tr><td>2020</td><td>CVPR</td><td>ClusterFit [174]</td><td>clustering</td><td>CNN</td><td>cla,seg</td><td>-</td></tr><tr><td>2020</td><td>ICML</td><td>CPC v2 [146]</td><td>patch overlapping</td><td>CNN</td><td>cla,det</td><td>-</td></tr><tr><td>2020</td><td>CVPR</td><td>PIRL [162]</td><td>Jigsaw puzzles</td><td>CNN</td><td>cla,rec,dec</td><td>https://github.com/../PIRL</td></tr><tr><td>2020</td><td>CVPR</td><td>MoCo [163]</td><td>discrimination</td><td>CNN</td><td>cla,rec,dec,pos,seg</td><td>https://github.com/../moco</td></tr><tr><td>2021</td><td>ICLR</td><td>PCL [171]</td><td>clustering</td><td>CNN</td><td>cla,det</td><td>https://github.com/../PCL</td></tr><tr><td>2020</td><td>arXiv</td><td>MoCo v2 [164]</td><td>discrimination</td><td>CNN</td><td>cla,dec</td><td>https://github.com/../moco</td></tr><tr><td>2020</td><td>ICLR</td><td>SeLa [175]</td><td>self-labelling</td><td>CNN</td><td>cla,det,seg</td><td>https://github.com/../self-label</td></tr><tr><td>2020</td><td>ICML</td><td>SimCLR [166]</td><td>discrimination</td><td>CNN</td><td>cla</td><td>https://github.com/../simclr</td></tr><tr><td>2020</td><td>NeurIPS</td><td>SimCLR v2 [176]</td><td>self-distillation [160]</td><td>CNN</td><td>cla</td><td>https://github.com/../simclr</td></tr><tr><td>2020</td><td>ECCV</td><td>CMC [177]</td><td>view matching [178]</td><td>CNN</td><td>cla,seg</td><td>https://hobbitlong.github.io/CMC</td></tr><tr><td>2020</td><td>NeurIPS</td><td>InfoMin [179]</td><td>discrimination</td><td>CNN</td><td>cla,det,loc,seg</td><td>https://hobbilong.github.io/InfoMin</td></tr><tr><td>2020</td><td>NeurIPS</td><td>SwAV [167]</td><td>cropping</td><td>CNN,Transformer</td><td>cla,det</td><td>https://github.com/../swav</td></tr><tr><td>2020</td><td>NeurIPS</td><td>BYOL [165]</td><td>discrimination</td><td>CNN</td><td>cla,det,seg</td><td>https://github.com/../byol</td></tr><tr><td>2021</td><td>arXiv</td><td>MoCo v3 [180]</td><td>discrimination</td><td>CNN,Transformer</td><td>cla</td><td>-</td></tr><tr><td>2021</td><td>ICLR</td><td>ReLU [181]</td><td>discrimination</td><td>CNN</td><td>cla,rel</td><td>-</td></tr><tr><td>2021</td><td>ICLR</td><td>PCL v2 [171]</td><td>clustering</td><td>CNN</td><td>cla,det</td><td>https://github.com/../PCL</td></tr><tr><td>2021</td><td>CVPR</td><td>SimSiam [170]</td><td>discrimination</td><td>CNN</td><td>cla,det,seg</td><td>https://github.com/../simsiam</td></tr><tr><td>2021</td><td>ICML</td><td>DirectPred [182]</td><td>discrimination</td><td>CNN</td><td>cla</td><td>https://github.com/../ssl</td></tr><tr><td>2021</td><td>ICCV</td><td>DINO [151]</td><td>discrimination</td><td>CNN,Transformer</td><td>cla,seg</td><td>https://github.com/../dino</td></tr><tr><td>2021</td><td>arXiv</td><td>MoBY [183]</td><td>discrimination</td><td>CNN,Transformer</td><td>cla,det,seg</td><td>https://github.com/../Transformer-SSL</td></tr><tr><td>2021</td><td>NeurIPS</td><td>MST [184]</td><td>token prediction</td><td>CNN,Transformer</td><td>cla,det,seg</td><td>-</td></tr><tr><td>2022</td><td>ICLR</td><td>BEiT [185]</td><td>token prediction</td><td>Transformer</td><td>cla,seg</td><td>https://github.com/../beit</td></tr><tr><td>2022</td><td>CVPR</td><td>MAE [154]</td><td>reconstruction</td><td>Transformer</td><td>cla,det,seg</td><td>https://github.com/facebookresearch/mae</td></tr><tr><td>2022</td><td>CVPR</td><td>SimMIM [155]</td><td>reconstruction</td><td>Transformer</td><td>cla,det,seg</td><td>https://github.com.microsoft/SimMIM</td></tr><tr><td>2022</td><td>ArXiv</td><td>UM-MAE [158]</td><td>reconstruction</td><td>Transformer</td><td>cla,det,seg</td><td>https://github.com/implus/UM-MAE</td></tr><tr><td>2022</td><td>ArXiv</td><td>LoMaR [159]</td><td>reconstruction</td><td>Transformer</td><td>cla,det,seg</td><td>https://github.com/junchen14/LoMaR</td></tr><tr><td>2022</td><td>Arxiv</td><td>CAE [186]</td><td>reconstruction</td><td>Transformer</td><td>cla,det,seg</td><td>https://github.com/lxtGH/CAE</td></tr><tr><td>2023</td><td>AAAI</td><td>PeCo [187]</td><td>reconstruction</td><td>Transformer</td><td>cla,det,seg</td><td>-</td></tr><tr><td>2023</td><td>ArXiv</td><td>SAM [156]</td><td>reconstruction</td><td>Transformer</td><td>det,gen,seg</td><td>https://github.com/facebookresearch/segment-anything</td></tr></table>


1 下游任务类型：分类、识别、检测、定位、分割、聚类、修复、检索、生成、姿态估计、强化学习。


预训练主干网络的参数。它在无监督学习中使用的一系列标准迁移任务上展示了 SOTA 性能。

当谈到对比学习和聚类之间的联系时，SwAV [167] 利用了作为聚类中心的原型来帮助在预训练期间对样本对进行分类，而原型对比学习 (PCL) [171] 首先旨在弥合对比学习与聚类之间的差距。与作为 pretext tasks 学习低级表征的实例判别相比，聚类可以帮助编码更多的语义信息。然后更多基于语义的下游任务将从中受益。如图 12 所示，原型对比学习使用原型来替代 NCE 损失（公式 (15)）中生成样本的视图之一，这是 PCL 中提出的 ProtoNCE 损失。此外，PCL 也是一种基于软参数共享的方法，其中动量编码器按照公式 (14) 更新。

# 4.8 总结

本节广泛调查了 PFMs 在图像表征学习方面的最新进展，从早期设计用于自标记的 pretext tasks 到目前基于对比损失的 SSL。主要方法的流程得到了清晰的说明。我们希望本节能让即将进入该领域的研究人员对这个新领域和一些值得的研究方向有一个基本的了解。我们

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-12/925beae9-6fc7-404b-a575-971981ce47ad/68a251580a60aa9dbd88eb1a55921a8a9b26c1b25686ff7deb6044c04f604dab.jpg)

图 13：图信息补全 (GIC) 和图属性预测 (GPP)。

相信 PFMs 强大的泛化能力将通过“一次预训练，永远迁移”极大地减少训练计算开销。最近基于 transformer 的 PFMs 在目标数据集上逐渐优于传统的从头训练。这一发现将激发对这一激动人心的领域的进一步探索和研究。

# 5 面向图学习的PFMs

随着深度学习在图领域的发展，模型的参数（即图嵌入）开始迅速增加。因此，需要大规模的标注数据来训练模型，以避免欠拟合或过拟合。然而，构建大规模图标注数据集的成本过于主观、昂贵且耗时，特别是在需要专业知识和时效性的领域。虽然一些半监督方法暂时缓解了图嵌入模型对标签规模的依赖，但并未从根本上解决这一问题。近期，受CV和NLP中成功的启发，研究人员开始关注PFMs在图领域的应用。然而，对于大多数图而言，由于节点和边等信息的独特性质，直接获取大规模预训练数据具有挑战性。因此，近期的研究集中在利用图的属性、拓扑和社区的内在信息来增强节点特征的有效性。我们在表3中总结了图相关的PFMs。

# 5.1 基于图信息完成的学习

基于图信息完成（GIC）进行预训练的本质动机是掩盖输入图数据的部分信息，并基于未掩盖的图数据恢复被掩盖的信息，从而预训练图嵌入，如图13所示。类似的思想较早出现在图像和文本处理领域。例如，在图像处理中，通过恢复图像像素和颜色等信息来预训练图像编码器；在文本处理中，许多方法通过基于上下文词恢复句子中的部分信息来实现词嵌入和编码器的预训练。这些方法启发了图PFMs中图完成任务的设计。

其中，You等人[188]受图像修复的启发，首次提出通过移除目标节点的特征来覆盖它们，然后恢复/预测被掩盖节点的特征。为了



![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-12/925beae9-6fc7-404b-a575-971981ce47ad/069f213244e11e5eaa814bd4a179da56855bc632e58c38e3ddab6ec1fcae7368.jpg)



(a) 上下文一致性。



(b) 自身一致性。



图14：图一致性分析 (GCA)。



恢复/预测被掩盖的信息，GraphCompetion [188]通过向GCN提供未掩盖的节点特征（限于每个目标节点的二阶邻居的2层GCN）来实现。GraphCompetion预训练的目的是帮助模型更好地执行特征表示，并教会模型从上下文中提取特征。You等人[188]提出了属性掩盖任务（即AttributeMask），该任务随机掩盖节点属性，然后要求自监督模块重构被掩盖的属性。Jin等人[189]对图数据上的SSL进行了深入思考，并提出了边掩盖任务（即EdgeMask），试图开发不仅基于单个节点本身，而且基于图中两个节点之间连接的自监督成对策略。具体而言，EdgeMask随机掩盖一些边，然后要求模型重构被掩盖的边。简而言之，EdgeMask旨在帮助GNN学习局部连通信息。Hu等人[190]提出了一种PFM，该模型掩盖节点和边属性，然后基于邻接结构预测这些被掩盖的信息。

# 5.2 基于图一致性分析的学习

与上述关注图中单个元素的方法不同，图一致性分析（GCA）主要探索图中两个元素分布的一致性。具体而言，具有相似语义的两个元素的一致性应显著强于语义不相关的两个元素，这一特性可用于预训练图模型。根据一致性的判断对象，此类方法大致可分为以下三类。

上下文一致性 基于早期的同质性假设，大量图模型倾向于将上下文节点投影到语义空间中的相似位置。图中的这种上下文一致性也被应用于预训练图模型，该模型试图通过捕获上下文中节点的分布特征来调整节点表示，如图14(a)所示。

随机游走是获取上下文的一种有效方法。它可以通过设计多种游走策略来捕获上下文中不同视角的分布特征。DeepWalk [191]采用截断随机游走策略，将节点上下文表示为节点序列的形式。通过将NLP的思想引入网络嵌入模型，DeepWalk将节点序列视为“句子”，并基于skip-gram模型对其进行建模，为节点表示提供了一种无监督且可扩展的训练方法。此外，在DeepWalk的基础上，node2vec [192]使用两种不同的参数控制随机游走策略来获取偏移的节点序列，以充分捕获上下文信息。

与从上下文中随机采样节点不同，一些近期的方法直接考虑节点k阶邻居分布（作为正样本）与非相邻节点（作为负样本）之间的关系，并利用这一点来训练图模型。LINE [193]分别提出了一阶和二阶邻近度，从不同视角描述图中节点对之间的局部相似性，并利用其优化节点表示。同时，LINE使用负采样和边采样技术来优化二阶遍历和过度的训练存储开销。VGAE [194]引入变分自编码器来编码图结构数据，并通过GCN编码器和简单的内积解码器对节点一阶邻居进行建模。

自身一致性 在NLP和CV领域，对比学习作为一种有效的自监督机制，被广泛应用于模型的预训练。事实上，此类方法的内部比较机制基于原始图数据和增强图数据之间的互信息估计，以保持数据本身的一致性，如图14(b)所示。受对比学习的启发，一些研究开始在图模型中生成原始数据样本的增强样本。其中，来自同一原始样本的两个增强样本被视为正样本对，来自不同原始样本的两个增强样本被视为负样本对。

对于节点级任务，GCC [195]将预训练任务设计为网络内和网络间的子图实例判别。GCC还通过引入对比学习，增强了GNN学习内在且可迁移的结构表示的能力。具体而言，GCC通过带重启的随机游走从全图中采样子图作为增强，并人为设计位置节点嵌入作为节点初始特征。作为一种新颖的图表示学习模型，GCA [196]结合了图拓扑和语义方面的各种先验来实现自适应对比增强。具体而言，GCA设计了一种基于节点中心性度量的增强方案，以突出重要的连接结构，同时通过向特定节点添加噪声来破坏节点特征，从而引导预训练模型识别潜在的语义信息。

对于图级任务，一些研究试图引入更多样化的对比学习策略。其中，You等人[197]基于底层先验，将四种常见的图增强任务（即节点删除、边扰动、属性掩盖和子图采样）引入GL模型，并提出了一个统一的对比学习框架：GraphCL。同时，GraphCL深入讨论了数据增强在对比学习中的作用，并通过实验证明联合多种增强策略可以提高模型性能。

跨尺度一致性 与上述两种考虑同一尺度内元素一致性的方法不同，对比不同尺度的图数据元素（例如节点-子图）也可用于训练图模型。大多数此类方法具有最大化互信息[198, 199]的思想。具体而言，通常使用读出函数来获取图/子图的摘要，并可以使用Jensen-Shannon散度来计算MI估计器。

作为代表性方法，DGI [200]依赖于最大化局部表示和相应高级图摘要之间的MI，这两者均使用建立的图卷积网络架构导出，从而学习节点表示。为了在单个图上生成负样本，DGI通过随机打乱节点特征来破坏原始图，同时保持结构不变。类似地，Hassani和Khasahmadi提出了CMVRL [201]，它基于图扩散生成样本图的额外结构视图。样本图和规则视图一起被二次采样，通过两个共享的MLP学习节点表示和图表示，然后通过判别器提供的一致性损失实现对比学习。

SUBG-CON [202]从原始图中采样一系列上下文子图，并将它们输入到编码器以获得池化的中心节点和子图表示。对于指定节点，上下文子图表示为正样本，其他随机采样的子图表示为负样本。潜在空间的对比损失将强制编码器识别正样本和负样本，以便基于区域结构信息区分不同的节点。

# 5.3 基于图属性预测的学习

将图的属性和结构信息视为信息完成的目标，基于图属性预测（GPP）的预训练也可以用于以不同形式构建图模型。最常见的方法之一是通过探索图数据中的辅助属性来生成自监督信号，并将图属性预测任务作为图模型的预训练任务。根据预文本任务的不同设置，可以大致分为两类：属性回归和属性分类。

属性回归 (PR) 在图模型中，与上述GIC不同，属性回归主要关注挖掘图中更广泛的数值结构和属性属性之间的关系。具体而言，这类方法在图数据中提取更丰富的自监督信号，用于预训练图模型。

例如，与掩盖节点属性相似但不同的是，NodeProperty [189]的目标是预测图中每个节点的辅助属性，例如度、局部节点重要性和局部聚类系数。换句话说，NodeProperty用于鼓励GNN在优化特定下游任务的同时捕获更丰富的局部结构信息。具体而言，NodeProperty将节点度视为代表性的局部节点属性，即自监督信号，并将其他节点属性视为未来的工作。同时，NodeProperty强调，设计与局部节点属性相关的自监督预文本任务的直觉，是最终指导GNN的特征嵌入（即节点表示）保存这些信息，这依赖于节点属性信息与特定任务相关的假设。

属性分类 (PC) 与属性回归任务不同，属性分类任务通常通过基于图数据中的某种分布定义伪标签来实现，这是一种典型的自监督方法。其中，结构密度、节点属性的相似性以及局部和全局分布之间的差异是最常用的。我们将简要介绍此类方法在GL预训练中的应用。

在这些方法中，聚类是最常见且有效的伪标签来源。其中，M3S [203]设计了一种多阶段训练策略，利用图聚类的思想迭代训练图编码器，在样本极小的情况下利用虚拟标签实现扩大标注数据。You等人[188]进一步提出了两种预训练策略。其中，节点聚类基于属性聚类为节点分配$K$（超参数）个伪标签，并通过节点分类预训练节点表示。此外，You等人还基于拓扑密度假设提出了图划分。在图划分中，图的节点被划分为大致相等的$K$（超参数）个子集，以最小化连接子集间节点的边数，然后为节点提供伪标签。

除了聚类方法外，一些研究人员基于图数据的其他统计特征生成伪标签。例如，在分子领域，Rong等人[204]利用子图的分子键和相关统计信息来指导GNN学习上下文敏感属性 (CSP)，然后将其应用于预测。Rong等人[204]提出了基序预测 (MP) 任务，该任务可以表示为多标签分类问题，其中每个基序对应一个标签。具体而言，假设考虑分子数据中的$K$个基序。对于特定分子（抽象为图$G$），他们使用RDKit检测每个基序是否出现在$G$中，然后将其作为基序预测任务的目标。

# 5.4 基于掩码自编码器的学习

掩码自编码器（MAE）首次应用于MAGE [205]中，即用于图上自监督学习的掩码自编码器。遵循MAE [154]，MGAE在基于卷积的部分网络结构（无掩盖边）上运行。此外，MGAE的解码器被设计为建模锚边的头尾节点之间的互相关。经验结果表明，MGAE优于传统的图自编码器和图SSL方法。此外，GMAE [206]通过使用Transformer代替卷积并重构掩盖节点的特征而不是掩盖边来扩展了这一方法。除了经验上的改进外，MaskGAE [207]进一步为掩码图建模的潜在好处提供了理论依据。设计算法以适应各种复杂属性的图是一个有前途的方向。例如，为了解决异构图场景，HGMAE [208]提出了元路径掩盖和具有动态掩码的自适应属性掩盖，以便在复杂的图结构上实现有效且稳定的学习。此外，还开发了多种训练策略，包括基于元路径的边重构以结合复杂的结构信息、目标属性恢复以利用各种节点属性，以及位置特征预测以编码节点位置信息。除了处理更复杂的图结构外，如何提高MAE在图数据上的学习效率仍然是一个未解决的问题。

# 5.5 图数据上的其他学习策略

除了上述方法外，还有许多使用相对新颖或混合策略的预训练方法。例如，$\mathrm { C G ^ { 3 } }$ [209]通过设计半监督一致性损失来生成改进的节点表示，以最大化同一数据或同一类别的数据的不同视图之间的一致性。接下来，$\mathrm { C G ^ { 3 } }$使用与输入特征相关的图生成损失来提取数据特征与输入图拓扑之间的潜在确定性关系，作为SSL的补充监督信号。

基于注意力机制，Graph-Bert [210]通过在其局部上下文内采样的无连边子图来训练自身重构节点属性和拓扑结构。GMI [211]将传统的互信息计算思想从向量空间扩展到图域，并提出联合最大化特征互信息（节点嵌入与其邻居的原始特征之间）和边互信息（两个相邻节点的嵌入）以进行图表示学习。GPT-GNN [212]提出了一种自监督图生成任务来指导自身捕获图的拓扑和语义属性。GPT-GNN大致将图生成的可能性分为属性生成和边生成，以解开节点属性和图拓扑之间的内在依赖。

# 5.6 总结

在图模型中，由于传统的特征学习方法在特征学习过程中常伴随信息丢失，且考虑的信息相对片面，所获得的图表示相对粗糙并丢失了大量信息。人们开始关注图数据中数据和属性的分布作为自监督信号来预训练图模型，使其能够捕获更有价值的信息。通过将图中的节点、属性和边的分布转化为不同的预文本任务，并使用GNN进行建模，图模型可以充分拟合输入图的原始分布。在大量无监督或半监督场景中，此类预训练图模型已被证明有利于下游任务。此外，联邦训练大图模型[213]可能是构建预训练基础模型的一个有前途的解决方案。目前，随着对比学习策略的深入研究，一些工作已尝试将不同形式的对比学习应用于图模型的预训练。通过对上下文、自身和跨尺度的一致性分析，此类方法大大提高了预训练图模型在不同图上的性能。

表3：GL中PFMs总结。

<table><tr><td>Year</td><td>Conference</td><td>Method</td><td>Pretext Task</td><td>Encoder</td><td>Code</td></tr><tr><td>2014</td><td>KDD</td><td>DeepWalk [191]</td><td>GC-C</td><td>Shallow NN</td><td>https://github.com/phanein/deepwalk</td></tr><tr><td>2015</td><td>WWW</td><td>LINE [193]</td><td>GC-C</td><td>Shallow NN</td><td>https://github.com/tangjianpku/LINE</td></tr><tr><td>2016</td><td>NeurlIPS</td><td>VGAE [194]</td><td>GC-C</td><td>GCN</td><td>-</td></tr><tr><td>2016</td><td>KDD</td><td>node2vec [192]</td><td>GC-C</td><td>Shallow NN</td><td>https://github.com/aditya-grover/node2vec</td></tr><tr><td>2017</td><td>NeurlIPS</td><td>GraphSage [214]</td><td>GC-C</td><td>Shallow NN</td><td>https://github.com/williamleif/GraphSAGE</td></tr><tr><td>2018</td><td>ICLR</td><td>DGI [200]</td><td>GC-CS</td><td>GCN/SAGE</td><td>https://github.com/PeterV-/DGI</td></tr><tr><td>2020</td><td>ICML</td><td>GraphCompetition [188]</td><td>GIC</td><td>GCN</td><td>https://github.com/Shen-Lab/SS-GCNs</td></tr><tr><td>2020</td><td>ICLR</td><td>AttMasking [190]</td><td>GIC</td><td>GCN</td><td>http://snap.stanford.edu/gnn-pretrain</td></tr><tr><td>2020</td><td>ICML</td><td>AttributeMask [188]</td><td>GIC</td><td>GCN</td><td>https://github.com/Shen-Lab/SS-GCNs</td></tr><tr><td>2020</td><td>arXiv</td><td>EdgeMask [189]</td><td>GIC</td><td>GCN</td><td>https://github.com/ChandlerBang/SelfTask-GN</td></tr><tr><td>2020</td><td>arXiv</td><td>NodeProperty [189]</td><td>GPP-PR</td><td>GCN</td><td>https://github.com/ChandlerBang/SelfTask-GN</td></tr><tr><td>2020</td><td>AAAI</td><td>M3S [203]</td><td>GPP-PC</td><td>GCN</td><td>-</td></tr><tr><td>2020</td><td>ICML</td><td>Node Clustering [188]</td><td>GPP-PC</td><td>GCN</td><td>https://github.com/Shen-Lab/SS-GCNs</td></tr><tr><td>2020</td><td>ICML</td><td>Graph Partitioning [188]</td><td>GPP-PC</td><td>GCN</td><td>https://github.com/Shen-Lab/SS-GCNs</td></tr><tr><td>2020</td><td>NeurlIPS</td><td>CSP [204]</td><td>GPP-PC</td><td>GCN</td><td>-</td></tr><tr><td>2020</td><td>NeurlIPS</td><td>MP [204]</td><td>GPP-PC</td><td>GCN</td><td>-</td></tr><tr><td>2020</td><td>NeurlIPS</td><td>SELAR [215]</td><td>GC-C</td><td>GNN</td><td>https://github.com/mlvlab/SELAR</td></tr><tr><td>2020</td><td>KDD</td><td>GCC [195]</td><td>GC-S</td><td>GIN</td><td>https://github.com/THUDM/GCC</td></tr><tr><td>2020</td><td>NeurlIPS</td><td>GraphCL [197]</td><td>GC-S</td><td>GCN</td><td>https://github.com/CRIPAC-DIG/GCA</td></tr><tr><td>2020</td><td>ICML</td><td>CMVRL [201]</td><td>GC-CS</td><td>GCN</td><td>-</td></tr><tr><td>2020</td><td>ICDM</td><td>SUBG-CON [202]</td><td>GC-CS</td><td>GCN</td><td>https://github.com/yzjiao/Subg-Con</td></tr><tr><td>2020</td><td>ICLR</td><td>InfoGraph [216]</td><td>GC-CS</td><td>GCN</td><td>https://github.com/fanyun-sun/InfoGraph</td></tr><tr><td>2020</td><td>AAAI</td><td>DMGI [217]</td><td>GC-CS</td><td>GCN</td><td>https://github.com/pcy1302/DMGI</td></tr><tr><td>2020</td><td>arXiv</td><td>Graph-Bert [210]</td><td>Hybrid</td><td>Transformer</td><td>https://github.com/jwzhanggy/Graph-Bert</td></tr><tr><td>2020</td><td>WWW</td><td>GMI [211]</td><td>Hybrid</td><td>GCN</td><td>-</td></tr><tr><td>2020</td><td>KDD</td><td>Gpt-GNN [212]</td><td>Hybrid</td><td>GNN</td><td>https://github.com/acbull/GPT-GNN</td></tr><tr><td>2021</td><td>ICML</td><td>JOAO [218]</td><td>GC-S</td><td>GCN</td><td>https://github.com/Shen-Lab/GraphCL_Automated</td></tr><tr><td>2021</td><td>AAAI</td><td>CSSL [219]</td><td>GC-S</td><td>GCN</td><td>https://github.com/UCSD-A14H/GraphSSL</td></tr><tr><td>2021</td><td>PAKDD</td><td>GIC [198]</td><td>GC-CS</td><td>GCN</td><td>https://github.com/cmavro/Graph-InfoClust-GIC</td></tr><tr><td>2021</td><td>WWW</td><td>SUGAR [199]</td><td>GC-CS</td><td>GCN</td><td>https://github.com/RingBDStack/SUGAR</td></tr><tr><td>2021</td><td>ICML</td><td>GraphLoG [220]</td><td>GC-CS</td><td>GCN</td><td>https://github.com/DeepGraphLearning/GraphLoG</td></tr><tr><td>2021</td><td>WWW</td><td>SLICE [221]</td><td>GC-CS</td><td>GCN</td><td>https://github.com/pnll/SLICE</td></tr><tr><td>2021</td><td>WSDM</td><td>BiGI [222]</td><td>GC-CS</td><td>GCN</td><td>https://github.com/caojiangxia/BiGI</td></tr><tr><td>2021</td><td>WWW</td><td>GCA [196]</td><td>GC-S</td><td>GCN</td><td>https://github.com/CRIPTAC-DIG/GCA</td></tr><tr><td>2021</td><td>KDD</td><td>HeCo [223]</td><td>GC-CS</td><td>GCN</td><td>https://github.com/liun-online/HeCo</td></tr><tr><td>2021</td><td>AAAI</td><td>CG³ [209]</td><td>Hybrid</td><td>GCN</td><td>-</td></tr><tr><td>2021</td><td>ICLR</td><td>SuperGAT [224]</td><td>GC-C</td><td>GAT</td><td>https://github.com/dongkwan-kim/SuperGAT</td></tr><tr><td>2021</td><td>KDD</td><td>MoCL [225]</td><td>Hybrid</td><td>GNN</td><td>https://github.com/illidanlab/MoCL-DK</td></tr><tr><td>2022</td><td>ArXiv</td><td>MGAE [205]</td><td>Maksed Edge Reconstruction</td><td>GCN</td><td>-</td></tr><tr><td>2022</td><td>KDD</td><td>GMAE [206]</td><td>Maksed Node Reconstruction</td><td>Transformer</td><td>https://github.com/THUDM/GraphMAE</td></tr><tr><td>2022</td><td>Arxiv</td><td>MaskGAE [207]</td><td>Partial Maksed Node Reconstruction</td><td>Transformer</td><td>https://github.com/EdisonLeeeee/MaskGAE</td></tr><tr><td>2022</td><td>Arxiv</td><td>HGMAE [208]</td><td>Metapath Masking Reconstruction</td><td>Transformer</td><td>-</td></tr></table>

# 6 针对其他数据模态的 PFMs

随着 PFMs 的快速发展，除了文本、图像和图之外，PFMs 在语音、视频、文本图像和跨数据方面也进行了大量研究。此外，研究人员最近已开始研究包含所有上述三个领域的统一 PFMs。因此，在本节中，我们介绍一些其他先进的和统一的 PFMs。

# 6.1 针对语音的 PFMs

在语音领域，wav2vec [226] 通过在大规模无标签数据集上捕获上下文信息来获取语音表示，并通过噪声对比二分类任务在少量样本上进行微调，从而极大地提高了下游任务的性能。此外，vq-wav2vec [227] 和 wav2vec 2.0 [228] 在 wav2vec 的基础上提出了一种离散的无监督预训练方法，将原始连续语音信号离散化，从而使成熟的 NLP 社区中的方法可以迁移和应用。同时，许多研究尝试设计不同的机制，将语音预训练获得的表示作为初始输入，并将其应用于不同的任务，例如自动语音识别 [229, 228]、音素识别 [230] 和语音合成 [231]。特别是，口语理解的广泛应用促进了语音和文本的联合预训练研究。例如，SpeechBERT [229] 将 MLM 应用于语音和文本对，以对离散信息进行表示学习。与依赖大量标注数据进行联合预训练的 [232] 不同，SPLAT [233] 使用无标签语音数据预训练语音嵌入模块，并提出了一种基于序列对齐的、适用于标签级下游任务的标签级对齐方法。MusicBERT [234] 是一种专为处理音乐数据而设计的预训练模型。它是通过在包含超过一百万首歌曲的庞大符号音乐语料库上进行训练而开发的。为了利用符号音乐数据改进预训练过程，MusicBERT 采用了几种机制，例如 OctupleMIDI 编码和小节级掩码策略。Huang 等人 [235] 建议在输入数据中融入节拍结构，这使得 Transformers 能够更好地在节拍-小节-乐句层面识别音乐的层次结构。AudioTransformer [236] 是一种通过实施某些技术（如以前在卷积网络中使用的池化）来增强 Transformer 架构性能的模型。Verma 等人 [236] 展示了他们如何利用基于小波的多速率信号处理思想来改进 Transformer 嵌入并获得更好的结果。

# 6.2 针对视频的 PFMs

视频类似于图像的 RGB 特征和文本的序列信息。在自监督视频表示学习中的许多有意义的探索不仅可以在视频数据集上高效执行，而且可以泛化到其他领域的学习中。Odd-One-Out Networks (O3N) [237] 是一种旨在预测训练数据集中视频采样的真实子序列中奇数视频子序列的技术。实验通过为 O3N 使用不同的视频片段编码器来进行，以证明这种预训练设计的一致性改进。类似地，Shuffle and Learn [238] 旨在从视频中的帧序列中学习正确的时间顺序。然而，Kim 等人 [239] 设计了一种名为 Space-Time Cubic Puzzles 的新自监督任务来训练 3D CNN。该任务需要一个预训练的骨干网络来排列排列后的 3D 时空裁剪。下游任务的性能证明，在解决此类谜题时已经学习了有效的视频表示。

受图像中对比学习的启发，视频中的许多预训练模型也利用对比损失来学习用于下游任务的视频表示。Inter-Intra Contrastive (IIC) 框架 [240] 可以通过使用从不同视频生成的正负对来学习视频表示。具体而言，同一视频中的不同模态被视为正对，而来自不同视频的视频片段被视为负对。Temporal Contrastive Pretraining (TCP) [241] 是另一种基于 CPC 学习视频表示的对比方法。与现有的直接为视频生成未来帧的基于 GAN 的方法不同，TCP 可以预测视频未来帧的潜在表示，这更有利于长期预测。Sequence Contrastive Learning (SeCo) [242] 是一种新颖的方法，在基于序列顺序的任务中考虑了帧内和帧间实例判别。

# 6.3 针对多模态的 PFMs

文本和图像之间的多模态 PFM 可以分为两类：单流模型和跨流模型。单流模型指的是在模型开始时整合文本信息和视觉信息。跨流模型指的是文本信息和视觉信息分别由两个独立的编码模块编码。然后不同的模态信息通过相互注意力机制进行融合。

单流模型 VisualBERT [243] 同时将文本和图像输入模型，并使用 Transformer 的自注意力机制进行对齐和融合。文本的输入与 BERT 相同，图像的输入是 Fasters-RCNN 提取的图像特征。VisualBERT 也进行预训练，然后对特定任务进行微调。它采用两个预训练任务，即 MLM 和句子图像预测，确定输入句子是否描述了相应的图像。Unicoder-VL [244] 的结构非常类似于 VisualBERT，除了对图像的处理。Unicoder-VL 通过 Faster-RCNN 提取图像特征，并将该特征与图像位置编码映射连接到同一空间。它增强了图像标签预测任务，该任务预测图像的类别。VL-BERT [245] 的预训练任务与 Unicoder-VL 相同。VL-BERT 的图像输入包括四个部分：由 Fasters-RCNN 提取的图像区域特征、该区域在原始图像中的位置、位置编码、片段编码和 [IMG] 编码。

跨流模型 在 ViLBERT [246] 中，文本和图像模式首先被单独编码，它们的输出经过一个标准的注意力模块。该模块基于 Transformer 结构。尽管如此，在自注意力机制中，每个模块使用其查询来计算与另一个模块的值和键的注意力，以整合不同模块之间的信息。该模型在两个任务上进行了预训练。第一个任务是掩码任务，这与 BERT 相同。在图像方面，任务的目标是当区域图像被掩码时，模型输出的分类分布可以尽可能与用于提取区域特征的模型（如 Faster-RCNN）的输出分布一致。第二个任务是语言图像匹配任务。DALL-E 是由 OpenAI 开发的一系列深度学习模型，旨在从自然语言提示生成图像。第一版 DALL-E 使用基于 Transformer 的架构，类似于 GPT LMs 中使用的架构，来处理文本提示并生成类似图像的表示。该模型基于 GPT-3 在图像及其相关文本描述的数据集上进行训练。DALL-E 2 [247] 是改进版本，它采用对比语言图像预训练 (CLIP) [248] 来捕获图像文本对之间的语义关联，并采用 GLIDE 扩散模型 [249] 进行文本条件图像合成。此外，OpenAI 最近提出了 GPT-4。它是一个大规模的多模态模型，采用 RLHF，并在各种专业和学术基准上展示了人类水平的性能。

基于包含比以前单模态数据更多可用信息的多模态数据，因此这些模型的性能通过与基准数据集上的 SSL 结合而得到增强。Cross and Learn [250] 是第一种揭示跨模态信息作为替代监督源的方法，并通过结合 RGB 和光流模态的跨模态损失和多样性损失获得强大的特征表示。与现有的从跨域数据集中的单个任务学习特征表示的方法不同，Ren 和 Lee 等人 [251] 提出了一种新颖的深度多任务网络，以学习更具泛化性的视觉表示，从而克服领域差异，并进一步利用不同任务中的跨域信息。在该论文中，跨域数据集是由基于 GAN 的网络生成的真实和合成数据集，而多个任务是 RGB 图像中表面法线、深度和实例轮廓的预测。该模型通过从跨域多任务特征学习中学习通用的视觉表示，其表现优于以前任何基于单任务的 SSL 方法。Tian 等人 [252] 认为，强大的表示是从人类视角理解世界，对跨视图因素进行建模的表示。他们提出了对比多视图编码 (CMC)，通过最大化同一场景不同视图之间的互信息来学习视频表示。

# 6.4 用于代码生成的 PFM

使用 LLMs 进行代码生成涉及使用预训练的语言模型，根据所需程序的自然语言描述自动生成代码。这种方法通过减少手动编码的需求并允许开发人员专注于更高级别的任务，有可能极大地提高软件开发的效率。

该技术涉及在大量自然语言文本上训练大规模语言模型，然后在特定的编程任务上微调模型。通过输入代码的自然语言描述，模型可以生成语法和语义正确的代码片段。使用 LLMs 进行代码生成已应用于各种编程领域，包括 Web 开发、NLP 和数据分析。用于代码生成的模型包括 GPT-4、T5 和 Codex 等。例如，Andrei 等人 [253] 已经调查和评估了 Transformer 模型用于个性化代码生成的微调。具体而言，他们已经在为 Java 方法生成单元测试和学习针对特定软件项目进行个性化的领域，评估了各种个性化技术的有效性。Shailja 等人 [254] 评估了 LLMs 生成有用的 Verilog 的能力。为此，预训练的 LLMs 在从 GitHub 和 Verilog 教科书收集的 Verilog 数据集上进行了微调。然后构建一个评估框架，包括用于功能分析的测试台和用于测试针对不同难度问题生成的 Verilog 代码语法的流程。经过微调的开源 CodeGen LLM 已被证明优于当前领先的商业 Codex LLM。CodeGen [255] 是一组多达 16.1B 参数的 LLMs，可以处理自然语言和编程语言数据。此外，他们作为开源发布了训练库 JAX FORMER。他们的工作表明，该模型在 HumanEval 上的表现可以与以前的最先进的零样本 Python 代码生成相媲美，展示了训练模型的实际应用。Poesia 等人 [256] 的研究中介绍的 Synchromesh 采用了一种称为目标相似性调优 (TST) 的新颖方法，从训练库中检索少量示例。然后，Synchromesh 利用这些示例训练预训练的语言模型，并通过应用约束语义解码 (CSD) 生成程序。CSD 是一个通用框架，可以将输出限制为目标语言中的有效程序。在这项工作中，作者表明，CSD 和 TST 的结合使用显着提高了预测精度，并防止了运行时错误。

然而，使用 LLMs 进行代码生成仍然存在一些局限性，例如模型倾向于生成过于冗长或低效的代码，以及它们无法处理复杂的编程任务。尽管如此，该技术已显示出巨大的前景，并具有彻底改变软件开发行业的潜力。

# 6.5 SOTA 统一 PFMs

处理多种模态的 PFMs 正在出现巨大的融合，例如骨干架构、预训练任务和模型放大 [29]。因此，研究人员提出了许多统一的 PFMs。统一的 PFM 是一个在单模态和多模态数据上预训练的统一模型，具有一个或多个 Transformer 作为骨干，能够执行大量的下游 AI 任务，包括单模态任务和多模态任务。目前基于模型架构有三种类型的 SOTA 统一模型。我们将它们定义为单 Transformer 模型、多 Transformer 模型和组合 Transformer 模型。单 Transformer 模型是指仅有一个大规模 Transformer 作为其骨干的 PFM 模型，而多 Transformer 模型是指具有多个 Transformer 的 PFM 模型。组合 Transformer 模型是结合了单 Transformer 和多 Transformer 结构的 PFM 模型。

单 Transformer 模型 UNITER [257] 是一种用于联合图像文本嵌入的大规模 PFM，由图像嵌入器、文本嵌入器和多层 Transformer 组成。它首先使用图像嵌入器对图像区域的视觉特征和边界框特征进行编码，并使用文本嵌入器对 Token 和位置进行编码。然后，应用 Transformer 模块通过四个预训练任务学习图像和文本的可泛化上下文嵌入。使用预训练任务的条件掩码，而不是对两种模态应用随机联合掩码。选择六个视觉语言任务作为下游任务。

Uni-Perceiver [258] 是一个具有共享参数的单一连体模型，能够处理关于视觉和语言任务的不同模态。不同的任务输入和目标被编码为具有模态特定 Tokenizer 的统一 Token 序列，然后由模态无关的权重共享 Transformer 编码器解码到共享表示空间。任何感知任务都被建模为通过表示的相似性为每个输入找到最大似然目标。Uni-Perceiver 在单模态和多模态任务上进行了预训练。各种下游任务的评估结果表明，通过对 $1 \%$ 的下游任务数据进行提示调优，其性能接近 SOTA 方法。

Gato [259] 构建了一个单一的大型 Transformer 序列模型，该模型作为一个多模态、多任务、多实现的通用策略。它可以使用具有相同权重集的单一神经网络执行各种任务。Gato 在 604 个任务上进行了训练，其中不同类型的数据，例如图像、文本、本体感觉、关节扭矩以及其他离散和连续的观察和动作，被序列化为一个扁平的 Token 序列，进行批处理，并由 Transformer 处理。在部署期间，采样的 Token 根据上下文组装成不同的动作。

OFA [26] 是一个简单的序列到序列学习框架，具有统一的基于指令的任务表示，统一了各种任务。在预训练和微调阶段，OFA 不需要为下游任务添加额外的特定于任务的层来实现任务无关。模态无关的计算引擎是一个 Transformer，其约束是不向下游任务添加可学习的特定于任务或模态的组件。OFA 在小规模的图像文本对上进行预训练以实现跨模态任务，同时在单模态任务上获得高度竞争力的性能。

UNIFIED-IO [27] 是一个使用统一架构执行大型和多样化任务的序列到序列模型。UNIFIED-IO 是一个 Transformer 模型，其中编码器和解码器都由堆叠的 Transformer 层组成。统一架构不需要特定的任务或模态分支，这是通过将每个任务的输入和输出同质化为一系列离散词汇 Token 来实现的。它在视觉和语言领域的 90 多个多样化数据集上训练单一的基于 Transformer 的架构。UNIFIED-IO 是第一个在不微调的情况下执行各种任务并在 16 个多样化基准中产生强结果的模型。

BEiT-3 [29] 是一个用于语言、视觉和视觉语言任务的通用多模态预训练模型。BEiT-3 的巨大融合可以从三个方面看到，包括骨干架构、预训练任务和模型放大。它引入了一个共享的 Multiway Transformer 作为骨干网络，在单模态和多模态数据上执行掩码数据建模。为了处理不同的模态，每个 Multiway Transformer 块都有一个共享的自注意力模块和一个前馈网络池。它是一个包含 1.9B 参数的大型基础模型。实验结果表明，BEiT-3 可以在视觉和视觉语言任务上优于 SOTA 模型。

多 Transformer 模型 FLAVA [28] 是一个同时对准所有模态的对齐模型，旨在解决视觉和语言任务以及视觉语言任务。它利用通用的 Transformer 模型架构从单模态和多模态数据中学习强大的表示。图像编码器 Transformer 用于捕获单模态图像表示。文本编码器 Transformer 用于处理单模态文本信息。多模态编码器 Transformer 将编码后的单模态图像和文本作为输入，并整合它们的表示以进行多模态推理。在预训练期间，掩码图像建模 (MIM) 和 MLM 损失分别应用于图像和文本编码器。另一方面，在配对的图像文本数据上使用掩码多模态建模 (MMM) 和图像文本匹配 (ITM) 损失。对于下游任务，分类头分别应用于来自图像、文本和多模态编码器的输出，用于视觉识别、语言理解和多模态推理任务。FLAVA 在不同领域的 35 个任务上显示出良好的性能。一个明显的优点是与其他模型相比，它使用的数据集更小。

组合 Transformer 模型 UNIMO [260] 可以使用一个模型学习单模态和多模态，以实现鲁棒和可泛化的表示。它采用多层自注意力 Transformer 同时学习通用的文本和视觉表示，并通过跨模态对比学习 (CMCL) 将它们统一到相同的语义空间。CMCL 背后的主要思想是保持配对的图像和文本表示在表示空间中接近，同时保持非配对的表示远离。所有这些都由同一个统一模态的 Transformer 成对或单独编码，并提取图像和文本的表示来计算对比损失。

# 7 关于 PFMs 的其他高级主题

随着预训练模型参数数量的增加，预训练模型需要更多的内存和计算资源。这增加了 PFMs 的训练成本，并限制了它们在资源受限设备上的部署。因此，为了提高预训练模型的效率，PFM 从以下两个方面提高计算效率：模型效率和模型压缩。PFM 的模型效率和压缩是指简化模型参数和结构的冗余。在不影响任务完成度的条件下，获得参数更少、结构更简洁的模型。

# 7.1 模型效率

模型效率致力于探索更高效的预训练方法，以较低成本的解决方案预训练大规模 PFMs。更高效的学习算法需要更有效的训练方法和更高效的模型架构。传统的预训练任务可能效率低下。例如，常用的掩码 Token 预测任务要求模型根据上下文预测掩码 Token。然而，样本中的掩码 Token 通常是输入 Token 的子集，模型只能从这部分 Token 中学习，因此训练效率低。为了解决这个问题，ELECTRA [30] 提出了一个 RTD 任务，预测每个输入标记是否被其他 Token 替换，这使得 ELECTRA 可以针对所有输入 Token 进行训练。除了有效的训练方法外，更高效的架构也可以提高 PFMS 的效率。对于大多数基于 Transformer 算法的 PFMS，可以通过降低 Transformer 算法的复杂性来获得更高效的模型架构。

# 7.2 模型压缩

模型压缩需要更少的计算资源和内存。这是一种减少模型大小并增强计算效率的潜在方法。模型压缩策略可以分为两种方式：参数压缩和结构压缩。

参数压缩的方法包括参数剪枝、参数量化、低秩分解和参数共享。参数剪枝是指为模型参数设计评估标准，以基于大规模的 PFM 删除冗余参数。例如，Compressing BERT [35] 在训练之前剪枝 BERT，同时保持与原始模型相当的性能。参数量化是将模型参数从 32 位全精度浮点数量化为低阶数。例如，Q8BERT [84] 使用 8 位量化将参数压缩四倍，而对模型性能影响很小。低秩分解是将高维参数向量的维度减少为稀疏的低维向量。参数共享是指利用结构化矩阵或聚类方法映射模型参数并减少参数数量。例如，ALBERT [36] 使用分解嵌入参数化和跨层参数共享来减少模型中的参数。

结构压缩是指紧凑网络和知识蒸馏。紧凑网络意味着通过设计新的紧凑网络结构来减少参数和计算量。知识蒸馏是指通过使用软标签等将知识从较大的教师模型转移到较小的学生模型。例如，DistilBERT [261] 使用知识蒸馏方法压缩 BERT，将 BERT 模型的大小减少 $40 \%$，同时保留了 $9 7 \%$ 的语言理解能力。

# 7.3 安全和隐私

PFMs 中的安全风险、社会偏见和数据隐私已成为一个重要的研究课题。Qiu 等人 [5] 指出，深度神经网络可能会受到对抗样本的攻击，从而误导模型产生错误的预测。由于预训练模型具有良好的可移植性，它们已被广泛应用于 NLP、CV 和 GL。然而，发现预训练模型容易受到对抗样本的影响。对原始输入的微小编扰可能会误导预训练模型产生特定的错误预测。同时，有可能通过查询 PFMs 来恢复数据样本，这可能导致隐私泄露。

生成对抗样本 对抗样本起源于图像。图像的对抗样本通过不可见的变化很难识别。例如，只修改了图像的一个像素。人类不容易检测到这种干扰，但神经网络可以识别修改后的图像，这是对抗样本的最初目的。一些工作发现预训练的 LMs 在某些场景中很脆弱。Jin 等人 [262] 通过生成自然对抗样本成功攻击了 BERT、CNN 和 RNN 三个目标模型，这表明当前的语言处理模型在安全性方面仍有很大的改进空间。然而，由于 NLP 中语言的独特离散性，很难实现。特别是，文本中对抗样本的生成必须考虑语言特征，以确保样本的语法和流畅性在影响模型输出的同时不受损害。例如，[263] 使用对抗样本成功攻击了用于文本分类和蕴含任务的 BERT 模型的微调阶段。[264] 结合了基于义原的单词替换方法和基于粒子群优化的搜索算法来生成对抗样本。

模型缺陷 一些无关的人为因素也可能误导 PFM 做出错误的预测。例如，[33] 发现由于利用数据集中的错误统计信息，BERT 在推理任务中的性能受到限制，这通过破坏这种属性极大地影响了性能。[265] 定义了通用对抗触发器。当触发器连接到任何输入时，它可以诱导模型生成特定的预测。

后门攻击 仍然有许多方法可以利用后门攻击操纵预训练模型的预测结果。[266] 证明可以构建一种权重中毒攻击，其中注入了预训练权重。在微调阶段之后，后门暴露出来。攻击者可以通过注入任意关键字轻松操纵模型预测。[267] 表明 NLP 中的 PFMs 可以通过修改模型语料库来操纵。新词或现有词的“含义”可以通过改变其权重参数来控制。

防御攻击 人机协同方法 [31, 32] 已被提出并应用于生成更自然、高效和多样化的对抗样本。一些防御方法已被提出来防御此类攻击。[268] 设计了一个辅助异常检测分类器，并使用多任务学习程序来防御对抗样本。另一方面，PFM 中的一些缺陷可能会被迁移学习中的自定义模型继承，例如上述对抗性漏洞和后门。为了缓解这个问题，[269] 提出了一种相关的模型切片技术，以减少迁移学习期间的缺陷继承，同时保留来自 PFM 的有用知识。

PFMs 中的数据隐私 LLMs 和其他 PFMs 已在私有数据集 [270] 上进行训练。研究人员发现，通过查询大规模 LMs，可以恢复特定的训练样本。例如，对手可能会获得 IRC 讨论和个人身份信息。更糟糕的是，由于大型模型有如此多的参数，PFM 很容易记忆或学习私人信息，使得大型模型比小型模型更容易受到攻击。许多 PFMs（如 LLMs）已在私有数据集上进行训练。研究人员发现，通过查询 LLMs 可以恢复单个训练样本。例如，对手可以提取包括个人身份信息和 Internet Relay Chat (IRC) 对话的样本。更糟糕的是，由于大型模型的数十亿参数，PFM 很容易学习私人信息，使得大型模型比小型模型更脆弱。为了减少隐私泄露的风险，我们必须在所有 PFM 过程中（包括数据处理、模型训练、模型推理和系统部署）考虑隐私保护措施。

# 8 未来的研究挑战和开放性问题

PFM 可以避免从头开始训练模型，这是从弱 AI 到通用 AI 的突破。目前，由于 PFM 具有大规模参数、大量训练数据和高计算复杂性等特征，PFMs 仍存在许多技术挑战。我们从四个方面总结了 PFMs 的未来研究挑战：数据、基础、模型设计以及上游和下游任务。同时，我们指出了未来研究方向中的一些开放性问题。

# 8.1 数据方面的挑战

大多数预训练数据集都是针对单模态和单语言的。构建用于多模态、多语言和图数据的预训练数据集，对 PFMs 的发展非常重要。对于这些数据的特征，现有的技术挑战如下：

数据不足 与 NLP 和 CV 不同，除了少数分子和蛋白质网络中的可重用节点外，图数据中的大多数节点和边没有大量的无标签数据用于预训练。同时，图模型的预训练研究仍处于初始状态。此外，物联网 的数据将是巨大的，并包含丰富的物理世界信息。例如，惯性测量单元传感器数据可以捕获用户的社会活动信息 [271, 272]。理论基础、预任务的各种定义以及对比学习的增强设计都不完善，迫切需要补充新的研究。

多模态 PFM 已经对多模态 PFMs 进行了一些研究工作，例如文本和图像、文本和音频等。这些主要是两种模态之间的 PFMs。目前，多模态 PFMs 的学习需要新的多模态数据集，这需要建立不同模式之间的数据。因此，多模态数据集的构建也是一个亟待解决的问题。

多语言 PFM 解决了多种语言中的资源短缺问题，并有助于在 QA、文本摘要、低资源神经机器翻译等方面取得新的改进。然而，当前的 PFM 仍然是掩码 LM。为了提高多语言 LM 的性能，需要添加一些合适的新任务。此外，多语言词汇表比单语言词汇表大得多，导致要学习的模型参数急剧增加。

# 8.2 基础方面的挑战

对于 PFM，无论是“黑盒”还是“白盒”方法，理论基础对模型性能都至关重要。研究的基础主要包括理论基础、语义理解和可解释性探索。

缺乏理论基础 CV 中的 SSL 从 NLP 中学习经验。没有深刻的理论来支持各种尝试性实验，进一步的探索也没有手册可循。尽管有几种理论分析试图理解预训练的崩溃或学习表示的泛化能力，但缺乏理论基础仍然是笼罩在 SSL 头上的巨大阴影。

语义理解 预训练的 LM 是学习了语言的意义，还是仅仅依赖语料库学习？许多模型在各种数据集上表现良好，可以提取有用的信息，其中一些方法甚至超过了人类水平。然而，在领域数据集或相对较小的数据集上，性能却很差。模型无法达到更好的稳定性水平并匹配不同的下游任务。这意味着模型无法服务于人类语言使用的真正目的。

# 8.3 模型设计方面的挑战

PFMs 的大多数现有结构都是针对文本、图像和图尝试的。主要方法是增加数据、提高计算能力和设计训练程序以获得更好的结果。如何在数据、计算资源和预测性能之间取得权衡值得研究。

模型多样性 模型设计有很多尝试，例如 CV 领域中基于生成的模型。然而，基于 GAN 的方法并不流行，原因如下：1) 判别器学习了有意义的特征表示，但在训练期间被遗忘了 [273]；2) 模式崩溃导致生成器输出单一模式的样本以欺骗判别器。结果，尽管研究人员尝试将基于 GAN 的方法应用于 SSL 进行预训练，但判别器收敛的困难和生成器发散的困难阻碍了该领域的发展和进步。

模型压缩 随着 Transformer 的广泛应用和预训练模型显示出普遍的增长趋势，预训练模型的计算复杂性已成为关注的焦点。由于模型训练的巨大硬件要求等原因，高门槛使得研究人员难以从头开始训练。BERT-base 和 GPT-3 分别包含约 1.08 亿个参数和 1750 亿个参数。这不利于相关研究工作的发展。有一些关于预训练模型压缩的工作，例如 ALBERT 比 BERT-base 具有更少的参数和更好的效果。改进的模型仍然需要强大的计算设备，使其难以普遍应用。降低高计算成本是未来研究的主要挑战之一。

模型鲁棒性 尽管许多研究人员为预训练设计了不同的预任务，但主要问题仍然是如何设计鲁棒的预任务并在大规模计算之前判断性能。此外，如何公平地比较这些提出的方法也是一个大问题。至于 NLP，深度神经网络由于其线性特征，容易受到对抗性输入的攻击。尽管预训练模型在不同的 NLP 任务上表现良好，但大多数都基于深度神经网络，这些网络的鲁棒性通常较差。在 CV 中，诸如切割和旋转之类的操作不会改变图像的性质。相反，在文本中添加、删除和替换单词等操作可能会影响文本的语义。因此，如何提高 NLP 中 PFM 的鲁棒性是一个技术挑战。

模型抗攻击 PFMs 容易受到对抗样本的攻击，这很容易误导模型产生特定的错误预测。由于 NLP 领域语言的独特离散性，这很难处理。因此，当前的 PFMs 在模型抗攻击方面还有巨大的改进空间。

# 8.4 微调和提示方面的挑战

NLP、CV 和 GL 领域的预训练模型可以在大多数上游任务中取得良好的性能，但在微调和提示的下游任务中并不总是表现良好。如何在上游和下游任务上取得一致的结果仍然是 PFMs 的一个挑战。

饱和现象 Google Research [274] 观察到上游和下游任务性能之间的非线性关系。在上游任务上使用更多数据获得更高的训练精度并不总是会在目标下游任务上带来更好的性能。这一观察挑战了对预训练过程的最直观理解。即使在最极端的情况下，上游和下游的性能也是不一致的。

预任务 有太多的自监督任务，也称为预任务。预任务可用于任何下游任务，例如检测和分类。很难匹配预任务和下游任务之间的关系。

基于任务的图 图上的许多预训练都是基于任务图的。不同的任务构建不同的图，其中节点需要被重用。这使得无法像 NLP 和 CV 那样引入大量数据在图上进行预训练。

# 8.5 未来 PFMs 的开放性问题

首先，期望文本、图像、图和多模态预训练的巨大融合。在撰写本调查时，没有工作在其统一的 PFMs 中考虑图。所有的 SOTA 统一模型主要集中在语言、视觉和语言视觉任务上，而忽视了图在数据域中的重要性。其次，未来研究中统一 PFMs 的统一骨干架构将变得更加流行。可以看出，只有一个大规模 Transformer 作为其骨干的统一 PFM 模型，即单 Transformer 模型，比其他类型的统一 PFMs 更受研究人员关注。第三，期望统一的 PFM 在所有数据域（包括文本、图像、图和多模态）的所有不同任务中实现 SOTA 迁移性能。大多数统一的 PFMs 仅在单个数据域中表现突出，而在其他域中的性能不具有竞争力。BEiT-3 [29] 在视觉和视觉语言任务中向这个研究方向展示了一个很好的例子。此外，在 PFMs 中 RL 的使用方面，尽管 ChatGPT 在 NLP 中建立了里程碑，但 CV 和 GL 尚未发表重要的研究。期望未来在这方面有更多的工作。

# 9 结论

本调查主要总结了文本、图像和图领域的现有 PFMs。首先，我们介绍了 NLP、CV 和 GL 的基本组件。然后，我们提供了为这三个领域中的预训练而设计的现有模型的总结，并总结了有关模型结构的必要信息。此外，我们研究了一些关于 PFMs 的其他研究，包括其他先进的和统一的 PFMs、模型效率和压缩以及安全和隐私。最后，我们介绍了 PFM 研究的主要挑战和开放性问题。

# A 基础组件

# A.1 NLP 的基础组件

表 4：NLP 和图中的常用符号。

<table><tr><td colspan="2">NLP</td><td colspan="2">Graph</td></tr><tr><td>Notations</td><td>Descriptions</td><td>Notations</td><td>Descriptions</td></tr><tr><td>N</td><td>输入文本的长度。</td><td>|·|</td><td>集合的长度。</td></tr><tr><td>wi</td><td>输入文本中的第 i 个词。</td><td>G</td><td>图的集合。</td></tr><tr><td>|V|</td><td>词汇语料库的大小。</td><td>G</td><td>一个图。</td></tr><tr><td>Hx</td><td>输入序列的表示。</td><td>V</td><td图中节点的集合。</td></tr><tr><td>θf</td><td>前向建模的参数。</td><td>v</td><td>一个节点。</td></tr><tr><td>θb</td><td>后向建模的参数。</td><td>E</td><td>图中边的集合。</td></tr><tr><td>θ</td><td>所有排列中的共享参数。</td><td>eij</td><td>vi 和 vj 之间的一条边。</td></tr><tr><td>ZN</td><td>T 的所有可能排列的集合。</td><td>A</td><td>图的邻接矩阵。</td></tr><tr><td>zT=t</td><td>z 的第 t 个元素。</td><td>T</td><td>图中节点类型的集合。</td></tr><tr><td>zT&lt;t</td><td>z 的 [1,2,...,t-1] 元素。</td><td>X</td><td>图的特征矩阵。</td></tr><tr><td>z</td><td>T 的一个排列。</td><td>Y</td><td>图中真实标签的集合。</td></tr><tr><td>m</td><td>特征向量的维度。</td><td>D</td><td>给定的图数据。</td></tr><tr><td>b1,b2</td><td>隐藏层和输出层的偏置值。</td><td>MGL</td><td>GL 模型。</td></tr></table>

# A.1.1 语言模型

随着深度学习的快速发展，LM 越来越多地应用于 NLP 模型的预训练。LM 可以估计一段文本的合理性概率。LM 主要有两种类型：统计 LM 和神经网络 LM。

统计 LM 统计 LM 是一种从概率和统计角度解决自然语言上下文相关特性的数学模型。统计 LM 的核心是确定一个句子在文本中出现的概率。作为概率 LM 的理论基础，N-gram 模型对后续 LM 产生了深远影响。它在 LM 领域发挥着举足轻重的作用。N-gram LM 引入了马尔可夫假设，该假设假设当前词出现的概率仅取决于最近的 $n - 1$ 个词。词 $w _ { i }$ 的最大似然概率可以通过下式计算：

$$
p \left(w _ {i} \mid w _ {1}, w _ {2}, \dots , w _ {N}\right) = p \left(w _ {i} \mid w _ {i - n + 1}, w _ {i - n + 2}, \dots , w _ {i - 1}\right) = \frac {C \left(w _ {i - n + 1} , w _ {i - n + 2} , \dots , w _ {i}\right)}{\sum_ {N} C \left(w _ {i - n + 1} , w _ {i - n + 2} , \dots , w _ {i - 1}\right)}, \tag {16}
$$

其中 $T = [ w _ { 1 } , w _ { 2 } , \dots , w _ { N } ]$ 是文本序列，$C ( w _ { i - n + 1 } , w _ { i - n + 2 } , \dots , w _ { i } )$ 是 $( w _ { i - n + 1 } , w _ { i - n + 2 } , \ldots , w _ { i } )$ 的共现频率。$p \left( w _ { i } \mid w _ { 1 } , w _ { 2 } , . . . , w _ { N } \right)$ 根据链式法则计算：

$$
p \left(w _ {1}, w _ {2}, \dots , w _ {N}\right) = \prod_ {i = 1} ^ {N} p \left(w _ {i} \mid w _ {1}, w _ {2}, \dots , w _ {i - 1}\right). \tag {17}
$$

N-gram 使用序列中每个词的概率来表示整个文本序列的共现概率。当 $N$ 较大时，表示对序列中下一个词的出现有更强的约束，并导致更稀疏的频率信息。当 $N$ 较小时，统计结果具有更高的可靠性和更好的泛化能力，但约束会较弱。

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-12/925beae9-6fc7-404b-a575-971981ce47ad/fab8522ede196dc77ca4bab5df07bb5c45f7ed99fab147660bef00beab294d5b.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-12/925beae9-6fc7-404b-a575-971981ce47ad/07977531360cefed1dd2e7795dacde79384816d410741a6d7147c1bc5ebd1f97.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-12/925beae9-6fc7-404b-a575-971981ce47ad/2f4016934139dce76c08a4e98c9f1e7b30160499ed4f29c2df75151190f16c96.jpg)


图 15：前馈神经网络、循环神经网络和预训练语言模型的模型架构。$H ^ { 1 , 2 }$ 、$H ^ { 2 , 3 }$ 和 $H ^ { 1 , 3 }$ 是用于连接每一层的权重矩阵。

神经网络 LM 统计 LM 采用最大似然估计，直观易懂。然而，仍然存在缺乏长期依赖、参数空间快速增长和数据稀疏等问题。因此，引入神经网络将 LM 映射到连续空间。神经 LM 使用词的分布式表示来对自然语言序列进行建模。与基于类的 N-gram 模型不同，神经语言模型能够识别两个相似的词，而不会丧失将每个词编码为彼此不同的能力。它可以直接用于 NLP 任务。它主要介绍了前馈神经网络 (FFNN)、循环神经网络 (RNN) 和预训练 LM。

如图 15 (a) 所示，FFNN 根据 $x = [ w _ { 1 } , \dotsc , w _ { i - 1 } ]$ 的所有先前词计算 $w _ { i }$ 的概率。为了预测 $w _ { i }$ 的条件概率，$x$ 根据投影索引共享投影矩阵 $M \in R ^ { | V | \times m }$ 到连续特征向量空间，$| V |$ 是词汇库大小，$m$ 是特征向量的维度。输出表示为：

$$
y = b _ {2} + H _ {x} ^ {1, 3} + H _ {x} ^ {2, 3} \tanh  \left(b _ {1} + H _ {x} ^ {1, 2}\right), \tag {18}
$$

其中 $H ^ { 1 , 2 }$ 、$H ^ { 2 , 3 }$ 和 $H ^ { 1 , 3 }$ 是用于连接每一层的权重矩阵，$b _ { 1 }$ 和 $b _ { 2 }$ 分别是隐藏层和输出层的偏置值。

FFNN 的结构仅包含有限的前文信息，对输入序列的长度有一定的限制。因此，RNN LM 应运而生。如图 15 (b) 所示，RNN 可以接受任意可变长度的输入。当移动输入窗口时，其内部状态机制可以避免重复计算，参数共享进一步减少了模型参数的数量。因此，与 FFNN 相比，RNN 具有很大的优势。

预训练 LM 是通过预训练某些任务来获得一组模型参数。它用这些参数初始化模型，然后进行训练以有效提高模型性能。常用的预训练模型有固定嵌入 (Word2vec [12], Glove [69] 等)、可变嵌入 (Embeddings from LMs (ELMO) [275], Generative Pretrained Transformer (GPT) [50] 和 Bidirectional Encoder Representations from Transformers (BERT) [13] 等)。这里，我们以 GPT 模型为例，如图 15 (c) 所示。它采用两阶段过程。在第一阶段，Transformer 解码器用作模型的基本单元来执行文本预测。在第二阶段，针对不同的下游任务对 GPT 进行不同的初始化，训练模型并微调参数。

# A.2 GL 的基础组件

由于图数据在许多领域的广泛使用，一些社区（例如，化学、蛋白质和社交网络）最近专注于图预训练的研究。这些预训练模型通过设计不同的代理任务，将图的属性、结构和其他信息从多个角度编码到节点表示中，用于优化下游任务。在本节中，我们介绍图的基本概念的定义，然后给出图上 PFM 的形式化定义。

# A.2.1 图的符号和定义

除非特别说明，本文中使用的符号如表 4 所示。我们使用 $\mathcal { G } = \{ G _ { i } \} _ { i } ^ { N }$ 来表示一组图，其中 $N$ 表示图的数量。根据图的边和节点的定义，图数据可以分为以下几种类型。

定义 1. 无属性图是 $G = ( V , E )$ ，其中 $v \in V$ 是一个节点，$e \in E$ 是一条边，自然地 $E \subseteq V \times V$ 。邻接矩阵 $A \in \mathbb { R } ^ { n \times n }$ 表示图 $G$ 的拓扑结构，其中 $n = | V |$ 。$A _ { i , j } = 1$ 表示节点 $v _ { i }$ 和 $v _ { j }$ 之间有一条边，否则 $A _ { i , j } = 0$ 。

定义 2. 属性图是 $G = ( V , E , X _ { v } , X _ { e } )$ ，其中 $\boldsymbol { X } _ { v } \in \mathbb { R } ^ { n \times d _ { v } }$ 和 $X _ { e } \in \mathbb { R } ^ { m \times d _ { e } }$ 分别是节点和边的特征矩阵，$| V | = n$ ，$| E | = m$ ，$d _ { v }$ 和 $d _ { e }$ 分别表示节点和边的特征维度。事实上，在大多数应用场景中，只有节点具有属性，边没有属性或只有权重。

定义 3. 无向图是 $G = ( V , E )$ ，其中 $e _ { i , j } \in E$ 意味着无序节点对 $( v _ { i } , v _ { j } )$ 。特别地，无向图的邻接矩阵 $A$ 是对称矩阵（即 $A _ { i , j } = A _ { j , i , }$ ）。

定义 4. 有向图是 $G = ( V , E )$ ，其中 $e _ { i , j } \in E$ 意味着有序节点对 $( v _ { i } , v _ { j } )$ 。

定义 5. $G$ 具有节点类型映射函数 $f _ { v } : V  T ^ { v }$ 和边类型映射函数 $f _ { e } : E \to \mathcal { T } ^ { e }$ 。当 $| \mathcal { T } ^ { v } | = | \mathcal { T } ^ { e } | = 1$ 时，图 $G = ( V , E )$ 是同构图。换句话说，$G$ 中的所有节点都属于一种类型，所有边也都属于一种类型。

定义 6. 当 $\vert \mathcal { T } ^ { v } \vert > 1$ 和/或 $| \mathcal { T } ^ { e } | > 1$ 时，图 $G = ( V , E )$ 是异构图。特别地，异构图必须是属性图。

# A.2.2 图上的学习设置

GL 方法通常用于解决图数据上的机器学习任务，我们介绍 GL 的不同设置（监督模式和学习模式）。

在此之前，我们首先提供 GL 相应数学公式的符号。$C =$ $\{ c _ { 1 } , c _ { 2 } , \cdots , c _ { K } \}$ 是在图集 $\mathcal { G }$ 中定义的一组目标组件 ( $G ^ { c _ { i } } \in \mathcal G )$ ，并且 $c _ { i }$ 与相应的真实标签 $y _ { i } \in \mathcal { Y } = \{ 1 , 2 , \cdot \cdot \cdot , N _ { y } \}$ 相关联，其中 $K$ 表示目标组件的总数，$N _ { y }$ 是预测的类别数。然后图数据可以表示为 $D = \{ c _ { i } , G ^ { c _ { i } } , y _ { i } \} _ { i } ^ { K }$ ，一个完整的 GL 模型 $M _ { G L }$ 也可以由 $y _ { i } = M _ { G L } ( c _ { i } , G ^ { c _ { i } } )$ 确定。例如，在节点分类任务中，$c _ { i }$ 是待分类的节点，$y _ { i }$ 表示 $c _ { i }$ 在图 $G ^ { c _ { i } }$ 中的标签。类似地，在节点聚类任务中，$c _ { i }$ 是待聚类的节点，$y _ { i }$ 表示图 $G ^ { c _ { i } }$ 中相应的聚类标签。

监督模式 根据训练数据的来源和规模，GL 的监督设置可以分为四种类型，如图 16 所示。监督 GL 是现实场景中最常见的模式。给定目标组件 $c _ { i }$ 和相应的真实标签 $y _ { i }$ ，目标是最小化 GL 模型的预测标签（即 $y _ { i } ^ { p r e d } = M _ { G L } ( c _ { i } , G ^ { c _ { i } } ) )$ 与所有 $c _ { i }$ 的期望标签 $y _ { i }$ 之间的损失函数。与监督学习相比，无监督 GL 指的是没有提供标签数据的情况，只能使用图数据的属性和结构分布（即 $( c _ { i } , G ^ { c _ { i } } ) )$ 。自监督 GL 是监督学习和无监督学习的特殊情况。具体而言，自监督学习主要使用代理任务（例如，聚类、补全和分区）从大规模无监督图数据中挖掘自己的监督信息（即伪标签），并通过自监督信息训练 GL 模型 $M _ { G L }$ ，使其能够学习到下游任务的有价值特征。换句话说，自监督学习的监督信息不是人工标注的，而是代理任务自动从大规模无监督数据中构建监督信息用于监督学习或训练。半监督学习是无监督学习和监督学习的结合，旨在学习数据分布以预测未标记数据，从而解决现实场景中获取标记数据困难的问题。在 GL 中，半监督学习是指在给定少量标记数据和大量未标记数据的情况下实现模式识别。

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-12/925beae9-6fc7-404b-a575-971981ce47ad/3b2a269afc9c05a47226749ba71e240315cf6aca1d58ef61a637ae19e33f2cba.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-12/925beae9-6fc7-404b-a575-971981ce47ad/7e73e35c78a96543c032fda05d44a12fbba64ebf46d235ae63e859f4d294d9a2.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-12/925beae9-6fc7-404b-a575-971981ce47ad/144b38226d3bddedbe00f2b1f69bd32bc38bebace0816057435bf44611805643.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-12/925beae9-6fc7-404b-a575-971981ce47ad/b4b56b3e5c62e96cc094da92ea8403eedf5891dabb3bdfd6181646a834ee768d.jpg)


图 16：不同监督模式的示意图。

学习模式 GL 模型 $M _ { G L }$ 通过给定的训练样本进行优化，并在验证样本上进行调整以参与测试。根据不同阶段图数据的可见性，GL 模型 $M _ { G L }$ 的学习设置可以分为两类：归纳式学习和直推式学习。

定义 7. 归纳式学习，这是机器学习任务中最常见的设置，在标记数据上训练模型，然后在训练阶段从未出现的样本上进行测试。形式上，给定训练样本 $\{ ( c _ { i } , G ^ { c _ { i } } , y _ { i } ) \} _ { i = 1 } ^ { N _ { l } }$ ，$\{ ( c _ { j } , G ^ { c _ { j } } ) \} _ { j = 1 } ^ { N _ { u } }$ ，其中 $N _ { l }$ 和 $N _ { u }$ 是标记/未标记样本的数量。归纳式学习学习一个函数 $f ^ { i n d } : \mathcal { G } \mapsto \mathcal { V }$ ，使得期望 $f ^ { i n d }$ 在未来的图数据 $\{ ( c _ { k } , G ^ { c _ { k } } ) \}$ 上是一个好的分类器，超出了 $\{ ( c _ { j } , G ^ { c _ { j } } ) \} _ { j = 1 } ^ { N _ { u } }$ 的范围。

定义 8. 直推式学习与归纳式学习不同，因为所有样本在训练和测试阶段都是可见的。形式上，给定训练样本 $\{ ( c _ { i } , \bar { G } ^ { c _ { i } } , y _ { i } ) \} _ { i = 1 } ^ { N _ { l } }$ ，$\{ ( c _ { j } , G ^ { c _ { j } } ) \} _ { j = 1 } ^ { N _ { u } }$ ，直推式学习学习一个函数 $f ^ { t r a n s } : \mathcal { G } ^ { l + u } \mapsto \mathcal { V } ^ { l + u }$ ，使得期望 $f ^ { t r a n s }$ 在未标记数据 $\{ ( c _ { j } , G ^ { c _ { j } } ) \} _ { j = 1 } ^ { N _ { u } }$ 上是一个好的分类器。

在监督设置（包括半监督/自监督）下，归纳式学习和直推式学习的统一分类器优化方法可以写成：

$$
\mathcal {L} = \frac {1}{K} \sum_ {i = 1 } ^ {K} \mathcal {L} \left(f _ {\theta} ^ {(\cdot)} \left(c _ {i}, G ^ {c _ {i}}\right), y _ {i}\right), \tag {19}
$$

其中 $\mathcal { L }$ 是交叉熵损失，$c _ { i }$ 可以是其关联图 $G ^ { c _ { i } }$ 的节点、边或子图，$f _ { \theta } ^ { ( \cdot ) }$ 表示带参数 $\theta$ 的归纳式/直推式函数。

与仅使用一个代理任务相比，一些方法设计了一些集成机制，将多个代理任务的优势合并到一个统一的框架中。

# B 传统学习方法

# B.1 传统文本学习

NLP 是一个融合语言学和计算机科学的研究领域。其主要研究任务包括词性标注、命名实体识别、语义角色标注、机器翻译、问答、情感分析、文本摘要、文本分类、关系抽取、事件抽取等。LM 可以被视为下游 NLP 任务的基石。它经历了四个过程：语法规则 LM、概率 LM、神经网络 LM 和预训练 LM。PFM 在大型基准数据集上进行训练以获得可以解决新的类似任务的模型，这已成为当前 LM 研究的一个新热点。

词表示在下游任务中起着重要作用，是 NLP 的基础。N-gram 模型对文本特征进行预处理并将相邻的 $N$ 个词编码为一组，这使其过度依赖于训练语料库的丰富性。否则，很可能会发生数据稀疏，并且计算复杂度会随着 $N$ 的增加呈指数级增长。神经网络 LM (NNLM) [11] 首次采用了词向量的思想，分布式表示的低维词向量可以很好地解决词嵌入引起的离散问题。然而，解决高计算复杂度的问题仍然具有挑战性。word2vec 模型的计算复杂度与所选窗口大小无关，而是由字典大小和词向量维度决定的。通过在初始训练后使用词向量嵌入在大语料库上进行训练，许多下游任务可以得到显著改善。然而，静态词向量的多义词问题仍未解决，它仍然属于浅层 LM [276] [277]。因此，迫切需要更有效的模型来更灵活地处理数据集。为了捕捉上下文的高级概念，如多义词消除、句法结构等。Neelakantan 等人 [278] 提出为每个词类型学习多个嵌入。Zhou 等人 [279] 整合矩阵两个维度上的特征，利用子词信息来丰富语义。基于 word2vec 中的连续词袋模型 (CBOW) [12]，Hui 等人 [280] 对生成的词向量进行情感微调，获得包含语义含义和情感倾向的词向量，显著提高了微博情感分类任务的性能。Liu 等人 [281] 提出了一种用于机器翻译的分层翻译模型。它使用基于 RNN 的神经 LM 作为词向量生成模型。Liang 等人 [282] 提出了一种基于双层自注意力机制的机器阅读理解方法，该模型分为三部分：单文档编码器、多文档编码器和答案预测。在单文档编码器中，通过门控循环单元 (GRU) 模型表示上下文信息的问题。Zhang 等人 [283] 提出了一种用于用户意图分类的独立 RNN (INDRNN) 和注意力机制，使用 word2vec 生成的词向量作为输入。该模型引入了词级注意力机制，有效量化了领域词汇对意图类别的贡献。

# B.2 传统图像学习

在深度学习时代，有几种类型的神经网络，从最著名的卷积神经网络 (CNN) 开始，到后来的基于注意力 和 Transformer 的网络。深度神经网络指的是具有更多隐藏层的人工神经网络，并且使用更多参数来表示目标模型，这使得其在从图像到视频的基准数据集上达到了 SOTA 性能。在这里，我们按时间顺序介绍 CV 中的里程碑网络。

# B.2.1 基于卷积的网络。

ImageNet [284] 作为计算机视觉中最重要的数据库之一，引发了许多图像分类中的里程碑网络架构，包括 AlexNet [285]、NIN [286]、VGG [287]、GoogLeNet [288]、ResNet [289]、DenseNet [290] 等。在目标检测和语义分割方面，研究人员在常见的基准数据集（例如 PASCAL VOC [309, 310]、MS COCO [311] 等）上探索了 R-CNNs [291, 292, 293, 294]、FCN [295]、SSD [296]、YOLOs [297, 298, 299, 300, 301]、SegNet [302]、PSPNet [303]、Deeplabs [304, 305, 306, 307]、RefineNet [308] 等。

这些流行的基于卷积的网络有几个共同特征：1) 数据增强。深度模型需要更多的数据来拟合复杂的模型，因此翻转、旋转、裁剪、缩放、平移甚至添加噪声等数据增强技术扩大了训练数据集；2) 卷积。卷积核用于提取原始图像数据的特征，保持了相邻像素的空间结构；3) 深度架构。深度架构包含更多参数，增强了模型的能力。这些共同特征促成了卷积神经网络 (CNN) 在近 10 年的计算机视觉中取得 SOTA 性能。

# B.2.2 循环神经网络

与针对 2D 维图像应用的 CNN 不同，循环神经网络 (RNN) [312, 313, 314] 尝试使用递归单元按顺序处理图片，即视频数据。然而，梯度爆炸和长期依赖的弱点限制了该模型的进一步发展。为了处理基于 RNN 的模型中嵌入的这些问题，Hochreiter 和 Schmidhuber 于 1997 年提出了长短期记忆 (LSTM) [315]。此外，LSTM 改进的能力产生了流行度，并在 NLP 和 CV [316, 317, 318, 319, 320] 中引起了关注。

# B.2.3 基于生成的网络

生成对抗网络 (GAN) [321] 提供了一种学习未标记数据表示的范式，并在下游任务中催生了许多基于 GAN 的方法。在图像翻译中，pix2pix 软件 [322] 首先提出了条件对抗网络作为图像到图像翻译问题的解决方案，并在现实世界数据集上取得了合理的结果。马尔可夫生成对抗网络 (MGAN) [323] 是一种生成纹理合成的方法，可以应用于风格迁移和视频风格化。CycleGAN [324] 提供了一种学习算法，将原始图像从源域转换到目标域，而无需在数据集中包含用于监督学习的成对图像。StyleGAN [325] 是一种基于风格的生成器，作为传统 GAN 的替代架构。像素循环神经网络 (PixelRNN) [326] 旨在通过对颜色通道之间的完整依赖关系进行建模来完成图像。DiscoGAN [327] 旨在学习不同域之间的关系。

GAN 也为研究数据合成提供了一个新的方向，因为它完美地模拟了原始数据的分布。拉普拉斯金字塔对抗网络 (LAPGAN) [328] 使用级联卷积网络以从粗到细的方式生成图像。类似地，堆叠生成对抗网络 (SGAN) [329] 将变化分解为多个级别，并通过以自顶向下的方式堆叠多个 GAN 来逐步解决不确定性。

# B.2.4 基于注意力的网络

基于 CNN 在 CV 领域的成功，注意力模块被设计用于配备流行的 CNN。例如，SENet [330] 提出了一个通道注意力模块，在 ILSVRC2017 竞赛中获得了第一名。此外，CBAM [331] 沿通道和空间维度依次推断注意力图。许多创新工作，如 GCNet [332] 和 CCNet [333]，都受到这种软注意力机制思想的启发，在识别和分割任务的主要基准上优于传统的 CNN。特别地，自注意力机制 [334]，通过关注同一序列内的所有位置来计算一个位置与序列中所有实体之间的响应，被提出用于估计特征图中一个位置与其他位置的相关性。为了控制预期的实体并对序列中不同元素之间更复杂的关系进行建模，掩码自注意力和多头注意力 [38] 是在 transformer 时代提出用于替代卷积功能的关键组件。

# B.2.5 基于 Transformer 的网络

最近，受到自注意力机制以及 Transformer 在 NLP 中后续成功的启发，CV 的研究人员也尝试使用 Transformer 作为卷积的替代品。基于自注意力的 transformer 模型通常在两阶段训练机制中运行：1) 通过定义代理任务在原始数据集（通常很大但标记不全）上进行预训练；2) 将预训练权重转移到下游任务，并通过微调调整目标域数据集上的参数。Vision Transformer (ViT) [40] 应用于 CV，并在主要基准数据集上达到了 SOTA 性能。数据高效图像 Transformers (DeiT) [335] 由 Facebook AI 提出，可以更高效地训练图像 transformers，同时保持 SOTA 性能。检测 Transformer (DETR) [336] 在目标检测和语义分割方面都显著优于竞争基线。LeViT [337] 在平衡准确率和训练速度方面优于现有基准。Image GPT [149] 受到 NLP 中序列 transformer 的启发，可以在 ImageNet 上与多个自监督基准竞争。基于这项研究，DeepViT [338] 探索了更深层次的架构，以通过使 transformer 更深来持续提高性能。此外，许多研究人员尝试将 transformer 应用于更具体的任务。金字塔视觉 Transformer (PVT) [339] 引入了金字塔结构来克服将 transformer 移植到各种密集预测任务的困难，并在主要基准数据集上达到了 SOTA 性能。M3DeTR [340] 是一项关于使用 transformer 进行多表示、多尺度和相互关系 3D 目标检测的新颖研究。医学 Transformer (MedT) [341] 专注于医学图像分割，并优于以前基于 CNN 和基于 transformer 的架构。总之，transformer 已成为 CV 中一个新颖且流行的研究领域，其性能已被许多现有工作证明。

# B.3 传统图学习

GL 旨在将图嵌入为低维表示，同时保留原始图数据的期望属性。经典的 GL 方法通常使用统计方法或人工设计的组件来实现。

降维 作为特征工程中常用的一种方法，降维旨在将高维属性图数据降低为低维表示。在 GL 中，它以丢失部分属性为代价来突出剩余信息。根据不同的降维策略，这些方法可以分为两类。第一类是线性假设下的子空间学习。基于与较大方差相关的主成分 [342] 代表重要结构信息，而较小方差代表噪声的假设，主成分分析计算最大化数据方差的低维表示。线性判别分析 (LDA) [343] 通过最大化类间散射和类内散射的比率来实现降维，从而获得线性投影矩阵。多维缩放 (MDS) [344] 是一种保持距离的流形学习方法。它在较低维度中产生一个映射，以尽可能保留节点之间的不相似性。第二类是非线性降维，其旨在自动学习非线性拓扑以实现流形学习。等距映射 [345] 首先在流形上构建邻域图并计算节点对之间的最短路径，然后使用 MDS 构建低维嵌入。局部线性嵌入 (LLE) [346] 首先为每个节点分配邻居。然后，它计算加权 $W _ { i , j }$ ，即来自其邻居的最佳线性重构特征 $X _ { i }$ 。最后，计算 $W _ { i , j }$ 的最佳重构的低维嵌入。

矩阵分解 深受降维思想的影响，基于矩阵分解的模型出现在 GL 的早期研究中。此类模型旨在重构图的邻接矩阵，以在保持结构信息的同时实现降维。尽管这些模型有显著的局限性，但实际上，它们的思想仍然启发着许多当前的研究。根据矩阵的构造方式，此类方法通常附加特定的约束。图拉普拉斯特征映射 [347] 最小化损失函数，以确保流形上彼此靠近的节点映射到低维空间中并且仍然保持局部距离。节点接近度矩阵分解 [348] 通过矩阵分解最小化目标函数 $| W - Y Y ^ { c T } |$ 以近似低维空间中节点的接近度，其中 $Y$ 和 $Y ^ { c }$ 是节点和上下文节点的嵌入，$W$ 是默认节点接近度矩阵。GraRep [349] 旨在保留图的高阶接近度到嵌入空间中，因此它通过将邻接矩阵自身相乘 $k$ 次来推导 $k$ 阶转移矩阵，$A ^ { k }$ 。从节点 $v _ { i }$ 到节点 $v _ { j }$ 的转移概率是 $k$ 阶转移矩阵的第 $i$ 行和第 $j$ 列中的条目，即 $p _ { k } ( v _ { i } | v _ { j } ) = A _ { i , j } ^ { k }$ 。然后 GraRep 使用 skip-gram 模型和负采样定义损失函数。为了捕获节点对之间的高阶接近度，HOPE [350] 在近似高阶接近度时保留了非对称传递性。具体而言，HOPE 的目标是最小化目标函数 $| | S - W C ^ { T } | | _ { F } ^ { 2 }$ ，其中元素 $s _ { i , j } \in S$ 表示相应节点对 $( v _ { i } , v _ { j } )$ 之间的特定边特征（例如，Katz 指数、根 PageRank、共同邻居和 Adamic-Adar），$W$ 是节点表示矩阵，$C$ 是作为上下文的节点的嵌入。为了更简单优雅地重构矩阵 $S$ ，HOPE 提出直接基于低秩奇异值分解 (SVD) 获得 $W$ 和 $C$ 。

图核 核方法是模式识别和机器学习中的重要算法。其基本思想是给出原始低维空间 $X$ 中的图嵌入 $x \in X$ ，并通过非线性函数 $f ^ { k e r }$ 将嵌入映射到高维特征空间 $H$ 。然后可以通过在 $H$ 中构造线性算法来解决 $X$ 中的非线性问题。图数据上的核方法主要有两种类型。第一类使用嵌入方法将图数据转换为向量表示，然后直接基于核函数实现应用。然而，由于在将图转换为向量表示时丢失了大量的图结构信息，此类方法在现实场景中表现不佳。第二类方法引入图核函数来解决这个问题。基于保留原始核函数的优势，它直接在高维希尔伯特空间中表示图数据的结构信息。传统图核方法的定义来自 R-卷积。根据对比子结构和图结构分解方法的不同，提出了大量基于图核的方法。例如，[351, 352] 的工作提出了一种基于计算两个图结构之间共同同步数量的随机游走核。为了降低计算复杂度并优化随机游走策略，提出了一种基于比较两个图结构之间最短路径信息的图核。为了捕获更复杂的拓扑信息，提出了 Weisfeiler-Lehman 子树图核，该核基于一维 Weisfeiler-Lehman 同构测试算法在一堆图结构中找到同构子树结构 [353]。

# C PFMs 理论

由于预训练受到了研究界的广泛关注，基于理论解释的研究同样引人注目。在 SSL 之前的无监督预训练时代，Erhan 等人 [354, 355] 对确认和理清学习困难的理论解释提供了一些见解。[354] 研究了预训练在架构深度、模型容量和训练样本数量方面的影响，并从优化和正则化的角度证明了预训练的鲁棒性。[355] 进一步证明了无监督预训练在下游监督任务中的正则化作用。

# C.1 不同视角

Pretext Tasks [356] 提出了一种基于近似条件独立性（CI）的机制，将代理任务和下游任务的数据分布联系起来，这表明代理任务可以从无标签数据中以自监督的方式学习表示，从而降低下游监督任务的样本复杂度。CV 和 NLP 任务上的实验都支持这一理论。Representation Learning via Invariant Causal Mechanisms (RELIC) [181] 也从显式的跨增强不变性约束可以产生改进的泛化保证这一角度提供了理论理解。

多视图冗余 从多视图设置的角度来看，[357] 将对比学习理解为利用数据的多个视图进行表示学习。该理论提供了一个分析，即来自预训练的这些表示的线性函数，与标签的非线性最优预测器相比，仍然具有竞争力。换句话说，只要不同的视图提供了关于标签的冗余信息，学习到的表示的线性函数在下游预测任务上几乎是最优的。

# C.2 不同类别

对比学习 虽然实验结果表明，以前的设计如对比损失或动量更新可以在 SSL 中产生令人印象深刻的性能。然而，SSL 中仍存在的最重要的问题之一是为什么这些方法可以在预训练过程中保持表示的一致性。一个朴素的观点是，正样本对之间的最小化可以促进不变性学习，而负样本对之间的最大化有助于避免表示坍塌。[358] 表明对比学习可以通过类内集中实现具有竞争力的界限，从而从迁移表示的收益中降低下游任务的样本复杂度。该研究还提供了一个框架，既可以用于保证预训练阶段学习表示的质量，也可以用于添加到框架中的未来假设，以允许更紧的保证。

非对比学习 虽然对比学习通过捕捉无标签示例之间的相似性和不相似性显示出效果，并进一步收敛到一个代表一般表示的平均局部最优，但最近的非对比 SSL 方法（如 BYOL 和 SimSiam）在没有负样本对比较设计的情况下也显示出 SOTA 性能。基于特征空间的分析，Tian 等人 [182] 研究了非对比 SSL 训练的行为，并证明了其效果来自于预测器和停止梯度信号。基于该理论，提出了一种新颖且简单的 DirectPred 方法作为该理论探索的副产品。

# D CV 上的代理任务分类

代理任务总是被设计为使用从数据本身生成的伪标签来预训练代理模型。自监督的代理任务有五类：1) 基于生成的方法；2) 基于变换的方法；3) 基于上下文的方法；4) 基于语义的方法；5) 基于视图的方法。

基于生成的方法 在深度学习时代，这类方法是基于 GAN 的。对于图像生成，有几个应用，包括图像着色 [138, 359]、图像超分辨率 [360]、图像编辑 [361]、上下文编码器 [137]、图像到图像翻译 [324] 等。另一方面，视频生成任务包含未来预测 [145]、视频动作识别 [241]、视频生成 [362, 363] 和视频表示 [364]。

基于变换的方法 变换是一种典型的技术，在传统深度学习中作为数据增强方法来扩大训练数据集。然而，如果同一图像的变换被标记为正样本，而其他被标记为负样本，这个代理任务就可以用于自监督预训练 [166]。自监督学习（SSL）中流行的变换包含颜色变换（如抖动、高斯模糊和调整亮度）和几何变换（如翻转、裁剪、缩放和旋转）。

基于上下文的方法 基本上，许多人工任务的设计和构造，例如解决拼图游戏 [140]、比较上下文相似性和区分序列顺序。解决拼图游戏被定义为识别图像中块的正确位置。该任务可以帮助模型学习一个用于迁移学习的编码器 [365, 141]，并且在预训练数据集足够大时，特征表示是有效的。此外，视频拼图的设计也被提出来用于无监督学习 [366]。不同的是，上下文相似性试图将来自同一图像的块标记为正样本，其他标记为负样本，然后使用预定义的相似性函数来缩放不同对之间的距离 [49]。

基于语义的方法 基于语义的方法包含目标检测、语义分割和深度预测。这些任务也涉及代理任务，因为它们基于像素的标签可以学习比更简单任务更鲁棒的特征表示。这些代理任务总是建立在视频数据集上 [367, 368]。

基于视图的方法 这类方法包含单模态数据和多模态数据。对于单模态数据，原始数据被视为锚点，不同的视角生成其正样本对。有时，基于序列的数据中的时间片被视为负样本对，因为场景随时间变化 [369]。此外，多模态数据在基于视图的方法中很常见，这里也被称为基于跨模态的方法。例如音视频协作学习 [370]、RGB 和光流跨模态距离训练 [250]。

# E 用于强化学习的 PFMs

预训练学习方法在监督学习领域的成功激发了人们研究 RL 领域的兴趣，即相同的范式是否可以适应 RL 算法。通用的预训练 RL 可以包括广泛的方向，例如 Reward-Free RL [371, 372, 373, 374]、Goalcondition RL [375, 376, 377] 和 RL 中的表示学习 [378, 379, 380, 381]。这里我们关注 RL 中的表示学习。具体而言，该方向旨在通过使用大规模数据集和无监督/自监督数据增强技术预训练 RL 智能体的视觉感知能力（即状态编码器）来提高性能。预训练过程使状态编码器能够从原始输入（CV 的像素级输入）中捕获基本的结构信息。基于预训练的状态编码器构建 RL 策略网络，以在微调阶段学习特定的下游控制任务。最近的研究表明，这可以从无监督 [382, 383, 384]、半监督 [385] 和自监督 [386, 387] 学习技术中大大受益于样本效率和学习有效性。具体而言，该方向大致可分为以下两类：基于模型的预训练 RL 和类对比预训练 RL。

基于模型的预训练 RL 基于模型的预训练 RL 旨在首先预训练一个生成式世界模型以捕获环境的潜在结构，然后在微调期间利用世界模型作为状态编码器或模拟器。World Models [382] 是第一项提出使用简单的变分自编码器以无监督方式学习环境的压缩空间和时间表示的工作，与从头开始训练相比，这大大提高了样本效率。然而，在不了解环境动态的情况下学习世界模型可能会导致忽略环境中的某些关键信息。Dreamer [388, 389] 提出通过近似表示、转移和奖励模型来学习潜在动力学。然后它们完全通过在潜在空间中的想象力来训练 RL 智能体，这更有效，因为它带来了较低的内存占用，并且能够快速并行预测数千个想象的轨迹。此外，DreamerPro [390] 提出了一种基于原型表示的无重建方法，以迁移潜在动力学建模中的任务无关视觉干扰问题。当存在复杂的背景干扰时，DreamerPro 显著优于以前的 SOTA 方法。为了验证为现实世界学习准确的世界模型是否有前景，Daydreamer [391] 将 Dreamer 应用于现实世界的物理机器人问题，并经验性地证明了显著的学习效率收益。

类对比预训练 RL 类对比预训练 RL 技术旨在通过使用大量域外数据预训练状态编码器或使用无监督学习或数据增强技术添加一些辅助损失来提高状态编码器的表示能力。CURL [392] 结合了实例对比学习并通过使用 MoCo [163] 机制，显著提高了 RL 智能体的数据效率。此外，RAD [393] 提出了一种隐式方法，直接在多个增强的观测视图上训练 RL 目标，这在 DeepMind Control Suite 的某些环境中优于 CURL。与 RAD 并发，DrQ [394] 引入了一个简单的正则化项，它应用图像增强来计算当前和目标 Q 值。他们表明，将其应用于 DQN 后，数据效率可以显著提高。DrQ-v2 [395] 通过将类似的技术插入 DDPG 算法，进一步扩展了该方法以解决复杂的人形运动任务。与该方向正交，[379, 378, 396, 397] 证明了在域外数据上使用监督或无监督方法预训练 RL 智能体的视觉部分可以提高下游 RL 控制任务的学习效率。除了确保不同观测视图之间的一致性外，SPR [381] 还训练了一个动力学模型，该模型强制表示具有时间可预测性。基于 SPR，SGI [380] 提出使用潜在动力学建模、无监督目标条件和逆动力学建模的组合来预训练表示。与以前的方法相比，SGI 可以更好地捕获环境的动态并促进下游 RL 控制任务的训练。

# F 评估指标

分类任务 分类任务根据标记的训练文档，确定文档特征与文档类别之间的关系。然后使用学习到的关系模型来确定新文档的类别。

准确率和错误率 文本分类模型的关键指标是准确率和错误率。准确率和错误率的术语定义如下：

$$
A c c u r a c y = \frac {(\mathrm {T P} + \mathrm {T N})}{N}, \tag {20}
$$

$$
\text {E r r o r R a t e} = 1 - \text {A c c u r a c y} = \frac {(\mathrm {F P} + \mathrm {F N})}{N}, \tag {21}
$$

其中 TP 和 FP 表示真阳性和假阳性，TN 和 FN 代表真阴性和假阴性。

精确率、召回率和 F1 除了标准类型和错误率，还有一些用于不平衡测试集的重要指标。这些指标与测试样本中类标签的概念相似。F1 被定义为 Precision 和 Recall 的调和平均值。因此，Accuracy、Recall 和 F1 可以表示为：

$$
P r e c i s i o n = \frac {\mathrm {T P}}{\mathrm {T P} + \mathrm {F P}}, \quad R e c a l l = \frac {\mathrm {T P}}{\mathrm {T P} + \mathrm {F N}}, \tag {22}
$$

$$
F 1 = \frac {2 \text {P r e c i s i o n} \times \text {R e c a l l}}{\text {P r e c i s i o n} + \text {R e c a l l}}. \tag {23}
$$

当准确率、F1 和召回率值达到 1 时，获得期望的结果。另一方面，当值变为 0 时，我们得到最坏的结果。对于多类分类任务，可以独立确定每个类的精确率和召回率值，然后可以分析单个和整体性能。

$M i c r o - F 1$ $M i c r o - F 1$ [398] 是衡量所有标签整体准确率和召回率的指标。我们将 $M i c r o - F 1$ 表示为：

$$
M i c r o - F 1 = \frac {2 P _ {t} \times R _ {t}}{P + R}, \tag {24}
$$

$$
P = \frac {\sum_ {t \in \mathcal {S}} T P _ {t}}{\sum_ {t \in \mathcal {S}} T P _ {t} + F P _ {t}}, \quad R = \frac {\sum_ {t \in \mathcal {S}} T P _ {t}}{\sum_ {t \in \mathcal {S}} T P _ {t} + F N _ {t}}. \tag {25}
$$

其中 $T P _ { t }$ 和 $F P _ { t }$ 表示文本上第 $t$ 个标签的真阳性和假阳性。

$M a c r o - F 1$ $M a c r o - F 1$ 通过给予所有标签相等的权重来计算所有标签的平均 $F 1$。$M a c r o - F 1$ 表示为：

$$
M a c r o - F 1 = \frac {1}{\mathcal {S}} \sum_ {t \in \mathcal {S}} \frac {2 \mathrm {P} _ {t} \times R _ {t}}{\mathrm {P} _ {\mathrm {t}} + \mathrm {R} _ {\mathrm {t}}}, \tag {26}
$$

$$
P _ {t} = \frac {T P _ {t}}{T P _ {t} + F P _ {t}}, \quad R _ {t} = \frac {T P _ {t}}{T P _ {t} + F N _ {t}}. \tag {27}
$$

其中 $T N _ { t }$ 和 $F N _ { t }$ 代表第 $t$ 个标签的阴性和假阴性。$\boldsymbol { \mathcal { S } }$ 代表所有样本的标签集。

平均倒数排名 (MRR) MRR 通常用于评估排序算法在问答（QA）和信息检索（IR）任务中的性能。MRR 表示为

$$
\mathrm {M R R} = \frac {1}{Q} \sum_ {i = 1} ^ {Q} \frac {1}{r a n k _ {i}}, \tag {28}
$$

其中 ranki 是第 $i$ 个真实答案的排名。每个文本上预测标签的数量用 $Q$ 表示。此外，还有一些指标，如 EM、Hamming-loss [399]、$\mathrm{ P @ { K } }$ 和 NDCG@K。

生成任务 生成任务使用语言模型（LM）根据输入数据预测下一个最可能的词或句子。

双语评估替补 (BELU) BLEU 将生成的句子与参考句子进行比较，并使用自动机器翻译算法进行预测。语言创建问题也得到了深度学习技术的支持，如语音识别、图像字幕生成和文本摘要。它们无法发现更好的东西，但它有几个优点：它易于理解，与人类判断密切相关，并且是独立于语言的。作为双语评估辅助工具，BLEU 主要用于评估机器翻译的质量 [400]。BLEU 比较候选文本中的 N-gram 与参考文本中的 N-gram 之间的重叠程度。重叠度越高表示翻译质量越好。计算公式为：

$$
B L E U = B P \times \exp \left(\sum_ {n = 1} ^ {N} W _ {n} \log P _ {n}\right), \tag {29}
$$

其中 $N$ 代表 N-gram，$BP$ 是惩罚因子，$P _ { N }$ 是多元精确率，$W _ { N } = 1 / N$ 是多元精确率的相应权重。$r$ 代表最短参考翻译的长度，$c$ 代表候选翻译的长度，那么惩罚因子 $BP$ 的具体计算方法如下：

$$
B P = \left\{ \begin{array}{l l} 1, & l _ {t} > l _ {a} \\ e ^ {1 - l _ {a} / l _ {t}}, & l _ {t} \leq l _ {a} \end{array} , \right. \tag {30}
$$

其中 $l _ { t }$ 是机器翻译中的词数，$l _ { a }$ 是参考答案中的词数。惩罚因子主要用于惩罚机器翻译和参考翻译之间的巨大差距。

ROUGE (面向召回率的文摘评估辅助工具) ROUGE 代表 N-gram 共现统计，用于自动评估方法。它是在 N-gram 的相似性上扩展的，这意味着 N-gram 是主文档文本中关于 $N$ 个词的子序列。ROUGE 有四种类型，包括 ROUGE-N、ROUGE-L、ROUGE-W 和 ROUGE-S。前两个是常用的，rouge-N 中的 $N$ 指的是 N-gram，其计算方式与 BLEU 类似，除了 BLEU 基于准确率，而 ROUGE 基于召回率。ROUGE-L 中的 $L$ 指的是最长公共子序列，计算为候选摘要和参考摘要之间的最长公共子序列。因此，长度越长，基于 $F$ 值的分数越高。主要介绍 ROUGE-N 和 ROUGE-L 的计算公式。ROUGE-N 的计算公式如下：

$$
R O U G E - N = \frac {\sum_ {S \in \{\text {R e f e r e n c e S u m m a r i e s} \}} \sum_ {\text {g r a m} _ {n} \in S} \text {C o u n t} _ {\text {m a t c h}} \left(\text {g r a m} _ {n}\right)}{\sum_ {S \in \{\text {R e f e r e n c e S u m m a r i e s} \}} \sum_ {\text {g r a m} _ {n} \in S} \text {C o u n t} \left(\text {g r a m} _ {n}\right)}, \tag {31}
$$

其中 $N$ 代表 N-gram，$C o u n t ( g r a m _ { n } )$ 代表 N-gram 的出现频率，$C o u n t _ { m a t c h } ( g r a m _ { n } )$ 代表 N-gram 的共现频率。ROUGE-L 的计算公式如下：

$$
R O U G E - L = F _ {l c s} = \frac {\left(1 + \beta^ {2}\right) R _ {\mathrm {l c s}} P _ {\mathrm {l c s}}}{R _ {\mathrm {l c s}} + \beta^ {2} P _ {\mathrm {l c s}}}, \tag {32}
$$

$$
R _ {\mathrm {l c s}} = \frac {L C S (X , Y)}{M}, \tag {33}
$$

$$
P _ {\mathrm {l c s}} = \frac {L C S (X , Y)}{N}, \tag {34}
$$

其中 $X$ 是候选摘要，$Y$ 代表参考摘要，$L C S ( X , Y )$ 表表示候选摘要和参考摘要的最长公共子序列（LCS）的长度，$M$ 代表参考摘要的长度，$N$ 代表候选摘要的长度。ROUGE 方法的特征是 N-gram 共现统计，基于召回率（ROUGE-N）和 F 值（ROUGE-L）。它们经常用于文本摘要。值得注意的是，ROUGE 是基于单词的对应关系，而不是基于语义的对应关系，但这可以通过增加参考摘要的数量来缓解。

METEOR METEOR，也称为显式排序翻译评估指标 [401]，是 BLEU 标准的改进版本，旨在解决 BLEU 标准中的一些缺陷。使用 WordNet 计算特定序列、同义词、词根、词缀和定义之间的匹配关系，提高了 BLEU 性能，并使其与手动判别更相关。计算公式如下：

$$
M E T E O R = (1 - P e n) \times F _ {\mathrm {m}}, \tag {35}
$$

$$
F _ {\mathrm {m}} = \frac {P R}{\alpha P + (1 - \alpha) R}, \tag {36}
$$

$$
P = \frac {m}{\sum_ {k} \mathbf {h} _ {k} \left(c _ {i}\right)}, \tag {37}
$$

$$
R = \frac {m}{\sum_ {k} \mathrm {h} _ {k} \left(s _ {i j}\right)}, \tag {38}
$$

其中 $\begin{array} { r } { P e n = \gamma ( \frac { c h } { m } ) ^ { \theta } } \end{array}$ 是惩罚因子，它惩罚候选翻译中与参考翻译不同的词序。ch 指的是块的数量，即候选翻译和候选参考翻译中彼此相邻的匹配单元的聚类单元。$\alpha , \beta , \theta$ 是可调参数，$m$ 是候选翻译中可以匹配的一元组的数量，$c$ 是候选翻译的长度，$h _ { k } ( c _ { i } )$ 是候选翻译 $c _ { i }$ 中的出现次数，$h _ { k } ( s _ { i j } )$ 是参考翻译 $s _ { i j }$ 中的出现次数。

困惑度 困惑度也称为混乱程度 [402]。其核心思想是：首先，根据测试句子，学习一个 LM $P$。然后，根据 LM $P$，计算可选句子的分数。最后，根据句子长度对上述分数进行标准化。计算公式如下：

$$
P P L (W) = P \left(w _ {1}, w _ {2}, \dots , w _ {M}\right) ^ {- \frac {1}{M}}, \tag {39}
$$

其中 $W$ 是候选翻译，$M$ 是候选翻译的长度，$P$ 是根据参考翻译获得的语言模型，$P ( w _ { 1 } , w _ { 2 } , \dots , w _ { M } )$ 是语言模型为候选翻译计算的分数。困惑度评估指标是基于语言模型的。困惑度越低，翻译质量越好，这通常用于机器翻译和语言模型。其缺点如下：数据集越大，困惑度下降越快；数据中的标点符号会影响模型的 PPL；以及常用词的干扰。

# G Datasets

# G.1 Downstream Tasks and Datasets on NLP

There are many available datasets in the NLP domain, divided according to different tasks. We summarize them in Table 5. It mainly comprises two categories: the task of classification of texts and the task of generating texts. The text classification tasks mainly include Sentiment Analysis (SA), News Classification (NC), Topic Labelling (TL), Natural Language Inference (NLI), Named Entity Recognition (NER), Question Answering (QA), Dialogue Act Classification (DAC), etc. The generation tasks mainly include text summaries and machine translation. 

Sentiment Analysis (SA) It consists of judging the emotional polarity and dividing it into several classes. Depending on the granularity of sentiments, the SA is divided into three categories: dichotomy (positive and negative), trichotomy (positive, negative, and neutral), and multiple categories. Here we introduce several datasets in detail. 

Stanford sentiment treebank (SST) [473] The dataset is an extension of MR [474]. SST-1 is a version of SST. It is divided into five categories and the number of training texts and testing texts is 8,544 and 2,210, respectively. It also consists of 20 average tokens. The SST-2 [475] contains 9,613 movie reviews including 6,920 training texts, 872 development texts, and 1,821 testing texts. 

Semantic textual similarity benchmark (STS-B) [476] It is used in semantic textual similarity tasks organized in the SemEval context between 2012 and 2017 [477]. It consists of text from image titles, news titles and forums. On a scale of 1 to 5, STS-B displays the semantic similarity of two sentences. It includes 5,749 training sets, 1,379 development sets, and 1,377 testing sets. 

Multi-Perspective Question Answering (MPQA) [478, 479] This is an opinion dataset which has two categories. It contains 10,606 sentences from various news sources that have been manually annotated for opinions and other private states. It is worth noting that there are 3,311 positive articles and 7,293 negative articles, having no labels for each article. 

IMDB reviews [480] The dataset is the world’s most authoritative source for binary sentiment classification of film reviews. The number of content in each class is the same and it can be divided into training and testing sets whose number of comments is 25,000 on average. 

News Classification (NC) As one of the most vital information sources, news content exerts a critical effect on people. The NC facilitates users to acquire essential knowledge in real time. Its applications mainly include news topic identification and recommendation of relevant news based on user interests. Here we introduce several datasets in detail. 


Table 5: The statistics of the datasets on NLP. For the QA task, the class represents the sum number of candidate answers and the correct answer. For dialogue, class is the number of slots. Length means the average tokens in turn.


<table><tr><td>Type</td><td>Task</td><td>Datasets</td><td>Class</td><td>Length</td><td>Number</td><td>Related Papers</td></tr><tr><td rowspan="35">Classification</td><td rowspan="5">Sentiment Analysis</td><td>MR</td><td>2</td><td>20</td><td>10662</td><td>[403, 404, 405, 406, 407]</td></tr><tr><td>SST-1</td><td>5</td><td>18</td><td>11,855</td><td>[408, 403, 409, 410, 411]</td></tr><tr><td>SST-2</td><td>2</td><td>19</td><td>9,613</td><td>[408, 403, 412, 413, 13]</td></tr><tr><td>MPQA</td><td>2</td><td>3</td><td>10,606</td><td>[414, 403, 415]</td></tr><tr><td>IMDB</td><td>2</td><td>294</td><td>50,000</td><td>[416, 417, 412, 413, 418, 14]</td></tr><tr><td rowspan="4">News Classification</td><td>20NG</td><td>20</td><td>221</td><td>18,846</td><td>[419, 420, 421, 406, 422, 279]</td></tr><tr><td>AG News</td><td>4</td><td>45/7</td><td>127,600</td><td>[423, 424, 425, 405, 14]</td></tr><tr><td>R8</td><td>8</td><td>66</td><td>7,674</td><td>[406, 422, 426]</td></tr><tr><td>R52</td><td>52</td><td>70</td><td>9,100</td><td>[406, 422, 426]</td></tr><tr><td rowspan="3">Topic Labeling</td><td>DBPedia</td><td>14</td><td>55</td><td>630,000</td><td>[423, 424, 418, 427]</td></tr><tr><td>Ohsumed</td><td>23</td><td>136</td><td>7,400</td><td>[406, 422, 426]</td></tr><tr><td>YahooA</td><td>10</td><td>112</td><td>1,460,000</td><td>[423, 428]</td></tr><tr><td rowspan="7">Natural Language Inference</td><td>SNLI</td><td>3</td><td>-</td><td>570,152</td><td>[429, 430, 55, 431, 13, 275]</td></tr><tr><td>MNLI</td><td>3</td><td>-</td><td>433,000</td><td>[432, 13, 14, 55, 36]</td></tr><tr><td>QNLI</td><td>2</td><td>-</td><td>115,667</td><td>[13, 14, 36]</td></tr><tr><td>WNLI</td><td>2</td><td>-</td><td>852</td><td>[431, 36]</td></tr><tr><td>RTE</td><td>2</td><td>-</td><td>5,768</td><td>[36]</td></tr><tr><td>SICK</td><td>3</td><td>-</td><td>10,000</td><td>[433]</td></tr><tr><td>MSRP</td><td>2</td><td>-</td><td>5,801</td><td>[434]</td></tr><tr><td rowspan="7">Named Entity Recognition</td><td>CoNLL 2003</td><td>4</td><td>-</td><td>2,302</td><td>[275, 13, 435, 436, 437, 438]</td></tr><tr><td>OntoNotes 4.0</td><td>18</td><td>-</td><td>-</td><td>[439, 440]</td></tr><tr><td>OntoNotes 5.0</td><td>18</td><td>-</td><td>2,945,000</td><td>[13, 435, 436, 438]</td></tr><tr><td>MSRA</td><td>3</td><td>-</td><td>-</td><td>[439, 13, 440, 438]</td></tr><tr><td>ACE 2004</td><td>7</td><td>-</td><td>443</td><td>[441, 442, 443, 444, 438]</td></tr><tr><td>ACE 2005</td><td>7</td><td>-</td><td>437</td><td>[441, 442, 443, 445, 438]</td></tr><tr><td>KBP2017</td><td>-</td><td>-</td><td>-</td><td>[445, 438]</td></tr><tr><td rowspan="6">Question Answering</td><td>QQP</td><td>2</td><td></td><td>799,266</td><td>[13, 36]</td></tr><tr><td>MRPC</td><td>2</td><td>-</td><td>-</td><td>[36]</td></tr><tr><td>SQuAD</td><td>-</td><td>5,000</td><td>5,570</td><td>[275, 55, 36]</td></tr><tr><td>RACE</td><td>5</td><td>-</td><td>100,000</td><td>[446, 14, 431, 36]</td></tr><tr><td>TREC</td><td>6</td><td>10</td><td>6,400</td><td>[404, 412, 425, 279, 405, 427]</td></tr><tr><td>WikiQA</td><td>-</td><td>873</td><td>243</td><td>[447, 448]</td></tr><tr><td rowspan="3">Dialog Act Classification</td><td>DSTC 4</td><td>89</td><td>-</td><td>30,000</td><td>[449, 450]</td></tr><tr><td>MRDA</td><td>5</td><td>-</td><td>62,000</td><td>[451, 449]</td></tr><tr><td>SwDA</td><td>43</td><td>-</td><td>1,022,000</td><td>[449, 452, 453]</td></tr><tr><td rowspan="12">Generation</td><td rowspan="4">Text Summarization</td><td>NYT</td><td>-</td><td>-</td><td>109,910</td><td>[454, 455]</td></tr><tr><td>CNN</td><td>-</td><td>760</td><td>92,579</td><td>[456, 457, 458, 459, 460]</td></tr><tr><td>Dailymail</td><td>-</td><td>653</td><td>219,506</td><td>[461, 457, 454, 462, 459]</td></tr><tr><td>Gigaword</td><td>-</td><td>-</td><td>3,991,000</td><td>[463, 457]</td></tr><tr><td rowspan="4">Machine Translation</td><td>WMT14</td><td>-</td><td>-</td><td>-</td><td>[464, 465]</td></tr><tr><td>WMT16</td><td>-</td><td>-</td><td>-</td><td>[466, 465]</td></tr><tr><td>WMT17</td><td>-</td><td>-</td><td>-</td><td>[467, 468, 466, 464, 469]</td></tr><tr><td>WMT18</td><td>-</td><td>-</td><td>-</td><td>[467, 466, 468]</td></tr><tr><td rowspan="4">Dialogue</td><td>DSTC2</td><td>-</td><td>-</td><td>3,000</td><td>[470]</td></tr><tr><td>MWOZ</td><td>35</td><td>15.03</td><td>10,438</td><td>[470, 471, 472]</td></tr><tr><td>GSIM</td><td>-</td><td>-</td><td>3,008</td><td>[470]</td></tr><tr><td>OOS</td><td>151</td><td>-</td><td>23,700</td><td>[470]</td></tr></table>

20 Newsgroups (20NG) [481] 20NG is a text dataset derived from newsgroups. There are 20 classes with the same number of articles per class, including 18846 articles in total. The average number of tokens is 221. 

AG News [423, 482] This is an academic news search engine, which is divided into four categories. It contains news headlines and introductions. It includes 120,000 training texts and 7,600 testing texts. The number of average tokens is 45/7. 

R8 and R52 [483] They come from Reuters [484]. R8 contains 8 classes consisting of 66 average tokens and includes 2,189 and 5,485 testing and training courses. There are 52 classes in R52, which consists of 70 average tokens. It is divided into 6,532 and 2,568 training and testing texts. 

Topic Labeling (TL) The task mainly obtains the meaning of the file by defining complex file themes. It is a critical component of topic analysis technology, which aims at simplifying topic analysis by assigning each article to one or more topics. Here, we introduce a few in detail. 

DBpedia [485] It is a large-scale multilingual knowledge base generated by Wikipedia’s most commonly used information boxes. It releases DBpedia every month, adding or removing classes and attributes in each version. The most popular version of DBpedia has 14 categories, separated into 560,000 training data and 70,000 testing data. The number of average tokens is 55. 

Ohsumed [486] This is a biomedical literature database. The number of texts is 7,400. It has 23 cardiovascular disease categories and consists of 136 average tokens. All texts are medical abstracts that are categorized into one or more classes. 

Yahoo answers (YahooA) [423] The dataset is a topic labeling task having 10 categories. The number of average tokens is 136. There are 140,000 training data and 5,000 testing data. Each text in YahooA has question titles, question contexts, and best answers. 

Natural Language Inference (NLI) This task is used to forecast whether the meaning of a text can be inferred from another. Interpretation is a broad form of NLI. By comparing the semantic similarity of sentence pairings, it determines whether a sentence is the interpretation of another one. Here we introduce several primary datasets in detail. 

The Stanford Natural Language Inference (SNLI) [429] It is commonly used in NLI takes. It contains 570,152 human-annotated sentence pairs, which are annotated with three sorts of relationships: neutral, derived, and conflicting. Multi-genre Natural Language Inference (MNLI) [487] has 3 categories and consists of 430,000 sentence pairs annotated with textual information, which is usually used in textual inference tasks. Question Natural Language Inference (QNLI) [488], whose task with 2 classes is to determine whether a given text pair is a question-answer. Winograd Natural Language Inference (WNLI) [489] which consists of 2 categories is a dataset that captures the standard reference information between two paragraphs. 

Microsoft Research Paraphrase (MSRP) [434] The dataset contains sentence pairs for the text-similarity task, including 1,725 training and 4,076 testing sets. A binary label annotates each pair, discriminating whether they are paraphrases. 

Sentences Involving Compositional Knowledge (SICK) [433] It includes nearly 10,000 English sentence pairs, marked with similarity, and the scale range is 1-5. It has neutral, entailment, and contradictory three categories. 

Named Entity Recognition (NER) This is a fundamental task of NLP to identify people, places, organizations, and other entities in text. It is a crucial primary tool for many NLP tasks, including information extraction, question answering, semantic parsing, machine translation, etc. 

CoNLL 2003 [275] It consists of newswire text from the Reuters RCV1 corpus. It contains four different entity types (Location, Organization, Person, and Miscellaneous) and includes 1,393 English news articles, and 909 German news articles. 

OntoNotes 5.0 [13] The dataset consists of 174,5K English, 900K Chinese, and 300K Arabic text data. It comes from telephone conversations, news agencies, radio news, radio conversations, and blogs. It has 18 entity classes containing 11 types, seven values, and 2,945,000 text data. 

MSRA [439] This is a Chinese dataset that is obtained from the news domain. It has three types of entities and is used as a shared task on SIGNAN back in 2006. 

Question Answering (QA) There are two types of QA systems: the extraction guidance system and the generation guidance system. The extractive QA can be regarded as a particular case of text classification. Here we detail several datasets. 

Microsoft Research Paraphrase Corpus (MRPC) [490] It contains 5,800 sentence pairs extracted from Internet news, and the task type is similar to the QQP dataset. Sentence pairs are derived from comments on the same news item and determine whether the two sentences are semantically the same. The assessment criteria were classification accuracy and F1 score. 

Stanford Question Answering Dataset (SQuAD) [275] This is a large-scale machine-reading comprehension dataset that contains two tasks. SQuAD 1.1 [488] provides questions and corresponding answers, and the dataset contains 100,000 samples in total, while SQuAD 2.0 [491] adds unanswered questions and expands the scale to 150,000. 

RACE [492] The dataset has 5 categories, containing nearly 100,000 questions extracted from middle and high school English tests, with corresponding answers given by experts. The average length of RACE text is more significant than 300, which is longer than other reading comprehension datasets (such as SQuAD) sequences. 

Dialog Act Classification (DAC) The dialogue act is a specific verbal component, which marks the dialogue according to the meaning category of the dialogue. DAC categorizes tags according to the meaning of the dialogue to help understand the speaker’s intentions. 

Dialog State Tracking Challenge 4 (DSTC 4) [450] It belongs to the dialog act classification task and mainly focuses on dialog state tracking on human-human dialogs. It is divided into 89 training classes and contains 24,000 training texts and 6,000 test texts. 

ICSI Meeting Recorder Dialog Act (MRDA) [451] It includes about 75 hours of speech from 75 naturally occurring meetings among 53 speakers. The number of categories is 5, and it contains 51,000 training texts, 11,000 test texts, and 11,000 validation texts. 

Switchboard Dialog Act (SwDA) [493] The dataset extends the dialogue behavior label with rounds/discourses. The label summarizes the sentence structure, and relevant and pragmatic information of the relevant turn. The SwDA is split into 43 training classes and includes 1,003,000 training texts, 19,000 test texts, and 112,000 validation texts. 

Text Summarization Text summarization is a summary of given single or multiple documents. It is kept as concise as possible while ensuring that it reflects the critical content of the original document. It can be divided into extractive summarization and generative summarization. Extractive summarization is generated by extracting and splicing the critical sentences in documents. Generative summarization is generated by a model, which summarizes documents according to the required content expressed in documents. 

NYT [454] The dataset comes from the corpus annotated by the New York Time. The named entities are annotated using the Stanford NER tool in conjunction with the Freebase knowledge base. It contains 9,076 articles, with the remaining 100,834 divided into a training set (96,834 examples) and a validation set (4,000 samples). 

CNN/Daily Mail [456] It is used for the passage-based question-answering task, and it is popular in assessing ATS systems. The dataset consists of CNN/Daily Mail news stories paired with multi-sentence human-generated summaries. There are 287,226 training instances, 13,368 validation instances, and 11,490 testing instances in total. 

Gigaword [463] This is a dataset of English news chapters consisting of nearly 950 pieces. Headlines – stories from multiple sources, including the New York Times – include some articles with a one-sentence, short news feed. 

Machine Translation (MT) It refers to the task of translation from one language to another with its semantic equivalence by a computer. There are three categories, rule-based machine translation, statisticsbased machine translation, and neural network-based machine translation. 

WMT14 [464] It is a grouping of datasets used in the Ninth Workshop on Statistical Machine Translation shared tasks, including a news translation task, a quality estimation task, a metrics task, and a medical text translation task. 

WMT16 [465] This dataset is a grouping of datasets used in the First Conference on Machine Translation shared tasks. It has ten shared tasks, including a news translation task, an IT domain translation task, a biomedical translation task, an automatic post-editing task, a metrics task, a quality estimation task, a tuning task, a pronoun translation task, a bilingual document alignment task, and a multimodal translation task. 

WMT17 [464] The dataset includes three MT tasks (news, biomedical, and multimodal), an automatic post-editing task, a quality estimation task, a task dedicated to the training of neural MT systems, a task on bandit learning for MT, an automatic post-editing task, and a metrics task. 

WMT18 [467] It mainly features six shared tasks: a news translation task, a biomedical translation task, an automatic post-editing task, a metrics task, a quality estimation task, and a multimodal translation task. Participants must evaluate their approaches to the machine translation topic using the standard datasets created for the shared tasks. 

Dialogue As an essential way of man-machine interaction, the dialogue system offers a wide range of applications. The existing dialogue systems can be grouped into task-oriented dialogue systems and nontask-oriented dialogue systems from application scenarios. Among them, the non-task type of conversation system can also be called a chatbot. 

DSTC2 [470] This is a multi-round dialogue dataset of restaurant reservation fields, including 1,612 training data, 506 verification data, and 1,117 test data. It allows the user’s goals to change compared to 

DSTC1. DSTC2 is also richer in terms of the conversation state representation, including the slot value pairs of the user’s targets and the ways to find them. 

MWOZ [470] It contains 8,420/1,000/1,000 conversations for training, validation, and test sets, respectively. It contains 30 pairs in seven domains being a multi-domain fully-labeled corpus. Every sample includes a goal, multiple user and agent utterances, and annotations regarding slot values. 

Out-Of-Scope (OOS) [470] The dataset includes 15,100 training, 3,100 validation, and 5,500 test sets, respectively. It contains 151 intent classes, containing 150 in-scope and one out-of-scope intent. The outof-scope intent indicates that a user utterance failed to classify to given predefined objectives. 

# G.2 Downstream Tasks and Datasets on CV


Table 6: The statistics of the datasets used on downstream tasks.


<table><tr><td>Type</td><td>Name</td><td>Usage</td><td>Domain</td><td>Class</td><td>Size</td><td>Related Papers</td></tr><tr><td rowspan="20">Classification</td><td>ImageNet</td><td>Pretrain &amp; Downstream</td><td>-</td><td>1000+</td><td>1,200,000+</td><td>[136, 137, 140, 141, 139, 142, 143, 138, 145, 494, 174, 179, 173, 151, 183, 182, 146, 153] [161, 162, 163, 164, 165, 48, 495, 49, 172, 166, 167, 170, 175, 177, 176, 180, 181, 496]</td></tr><tr><td>CIFAR-10</td><td>Downstream</td><td>-</td><td>10</td><td>60,000</td><td>[134, 135, 138, 165, 172, 175, 166, 173, 182]</td></tr><tr><td>CIFAR-100</td><td>Downstream</td><td>-</td><td>100</td><td>60,000</td><td>[165, 175, 166, 173]</td></tr><tr><td>STL-10</td><td>Downstream</td><td>-</td><td>10</td><td>6,000</td><td>[134, 135, 177, 179, 173, 182]</td></tr><tr><td>Caltech-101</td><td>Downstream</td><td>object</td><td>101</td><td>9,146</td><td>[134, 135, 165, 166]</td></tr><tr><td>MNIST-10</td><td>Downstream</td><td>digit</td><td>10</td><td>60,000</td><td>[48, 179]</td></tr><tr><td>SVHN</td><td>Downstream</td><td>digit</td><td>10</td><td>73,257</td><td>[175]</td></tr><tr><td>Places205</td><td>Downstream</td><td>scene</td><td>205</td><td>2,448,873</td><td>[138, 139, 142, 161, 162, 49, 172, 494, 175, 167, 173, 174, 496]</td></tr><tr><td>SUN397</td><td>Downstream</td><td>scene</td><td>899</td><td>130,519</td><td>[166]</td></tr><tr><td>HMDB51</td><td>Downstream</td><td>action</td><td>51</td><td>7000</td><td>[177]</td></tr><tr><td>UCF101</td><td>Downstream</td><td>action</td><td>101</td><td>-</td><td>[177]</td></tr><tr><td>Food-101</td><td>Downstream</td><td>food</td><td>101</td><td>101,000</td><td>[165, 166]</td></tr><tr><td>Birdsnap</td><td>Downstream</td><td>bird</td><td>500</td><td>49,829</td><td>[166]</td></tr><tr><td>Cars</td><td>Downstream</td><td>car</td><td>196</td><td>16,185</td><td>[166, 165]</td></tr><tr><td>Aircraft</td><td>Downstream</td><td>aircraft</td><td>102</td><td>10,200</td><td>[165, 166]</td></tr><tr><td>Pets</td><td>Downstream</td><td>pet</td><td>37</td><td>7,400</td><td>[165, 166]</td></tr><tr><td>Flowers</td><td>Downstream</td><td>flower</td><td>102</td><td>8,189</td><td>[165, 166]</td></tr><tr><td>DTD</td><td>Downstream</td><td>texture</td><td>47</td><td>5,640</td><td>[165, 166]</td></tr><tr><td>iNaturallist2018</td><td>Downstream</td><td>species</td><td>8,000+</td><td>450,000+</td><td>[162, 167, 174, 496]</td></tr><tr><td>JFT-300M</td><td>Pretrain</td><td>-</td><td>3,000+</td><td>300,000,000+</td><td>[40, 496]</td></tr><tr><td rowspan="2">Detection</td><td>COCO</td><td>Downstream</td><td>object</td><td>80</td><td>200,000</td><td>[142, 163, 164, 179, 167, 170, 183, 496]</td></tr><tr><td>VOC07</td><td>Downstream</td><td>object</td><td>20</td><td>9,963</td><td>[137, 138, 140, 139, 142, 143, 138, 146, 161, 162, 163, 164, 165, 49, 494, 175, 167, 170, 141]</td></tr><tr><td rowspan="7">Segmentation</td><td>VOC12</td><td>Downstream</td><td>object</td><td>20</td><td>2,913</td><td>[137, 138, 140, 139, 142, 49, 141]</td></tr><tr><td>NYU-Depth V2</td><td>Downstream</td><td>scene</td><td>894</td><td>1,449</td><td>[139, 165, 177]</td></tr><tr><td>VOC11</td><td>Downstream</td><td>object</td><td>20</td><td>3,334</td><td>[136]</td></tr><tr><td>ADE20K</td><td>Downstream</td><td>scene</td><td>3,688</td><td>27,574</td><td>[183, 496]</td></tr><tr><td>Cityscapes</td><td>Downstream</td><td>scene</td><td>25</td><td>25,000+</td><td>[163]</td></tr><tr><td>LVIS</td><td>Downstream</td><td>vocabulary</td><td>1,200+</td><td>160,000+</td><td>[163]</td></tr><tr><td>DAVIS</td><td>Downstream</td><td>scene</td><td>150</td><td>-</td><td>[151]</td></tr><tr><td>Inpainting</td><td>Paris StreetView</td><td>Downstream</td><td>scene</td><td>-</td><td>15,000</td><td>[136, 137]</td></tr><tr><td>Sequence</td><td>Moving-MNIST</td><td>Downstream</td><td>digit</td><td>-</td><td>10,000</td><td>[179]</td></tr><tr><td>-</td><td>YFCC100M</td><td>Pretrain</td><td>multimedia</td><td>-</td><td>100,000,000+</td><td>[49]</td></tr></table>

The datasets in CV mainly contain three types from the perspective of tasks: classification, detection, and segmentation. The popular datasets are concluded in Table 6, and some infrequently mentioned datasets in long tails are discussed in the text. 

Classification In this part, we first cover the popular large-scale datasets used frequently in both the pretext and downstream tasks. Then the domain datasets only used for the downstream tasks are unfolded. 

MNIST [497] It’s a collection of handwritten digits that includes 60, 000 samples in training and 10, 000 in testing. The images are fixed-size with $2 8 \times 2 8$ pixels. The pixel values are from 0 to 255.0 in which pixel values smaller than 255.0 can be understood as background (white) and 255 means foreground (black). The labels are from 0 to 9 and only one of these digits exists in an image. Both traditional and deep learning methods are based on this most popular dataset despite advanced methods showing perfect results. Thus, Geoffrey Hinton has described it as "the drosophila of machine learning". 

Street View House Numbers (SVHN) [498] In the domain of digit numbers, it collects real-world digit numbers from house numbers in Google Street View images. It includes 73, 257 digits for training, 26, 032 digits for testing, and 531, 131 additional. All of them are $3 2 \times 3 2$ color images with both class labels and character-level bounding boxes. 

CIFAR [499] As more advanced methods show perfect results on the simple datasets, more sophisticated datasets such as CIFAR-10 and CIFAR-100 are conducted. These two datasets are closer to the real-world object. The CIFAR-10 contains 50, 000 training images and 10, 000 testing images, with 6, 000 images per class and $3 2 \times 3 2$ pixels in each RGB color image. The CIFAR-100 is similar to the CIFAR-10 but with more detailed label information. There are 100 classes containing 500 training images and 100 testing images in each class. In addition, these 100 "fine" classes are grouped equally into 20 "coarse" classes. Researchers can adapt it to suitable learning methods. 

STL-10 [500] Inspired by the CIFAR-10 dataset, STL-10 is another $9 6 \times 9 6$ color image dataset containing similar 10 real-world classes. Each class has 500 training images and 800 testing images. The biggest difference is that STL-10 has 100, 000 unlabeled images for unsupervised learning. More construction information can be seen in [501]. 

Caltech-101 [502] It collects roughly $3 0 0 \times 2 0 0$ color images of objects belonging to 101 categories, with 40 to 800 images per category and 50 on average. The outlines of the objects in the pictures are annotated for the convenience of different learning methods. 

ImageNet [284] This is one of the most popular and large-scale datasets on computer vision. It is built according to the hierarchical structure of WordNet [503]. The full ImageNet dataset contains 14, 197, 122 images and 21, 841 synsets indexed, attaching on average 1, 000 images to demonstrate each synset. The most frequently-used subset of ImageNet is the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) dataset from 2010 to 2017, containing tasks of classification, localization, and detection. The number of samples in training and testing datasets and the labels of images are determined by the specific task, more details are seen in [504]. 

HMDB51 [505, 506] In addition to the popular MNIST, there still exist many domain datasets used for the downstream tasks in the classification problem. HMDB51 is an action video database for a total of 7, 000 clips in 51 action classes. It contains five types of facial actions and body movements. 

UCF101 [507] It is another action video dataset designed for more realistic action recognition. It is an extension of the UCF50 [508] dataset containing only 50 action categories with 101 action categories, collected from YouTube. What makes it a famous recognition dataset is the workshop in ICCV13 with UCF101 as its main competition benchmark. 

Food-101 [509] This is a real-world food dataset of 101 food categories, with 750 and 250 images per class in training and testing dataset respectively. 

Birdsnap [510] It is a fine-grained visual categorization of birds on a broad scale, with bounding boxes and the locations/annotations of 17 parts in the object. It contains 49, 829 images of the 500 most common species in North America, with each species containing 69 to 100 images and most species having 100. In addition, some images are also labeled as male or female, immature or adult, and breeding or non-breeding plumage. 

SUN397 To target the scene categorization, the extensive Scene UNderstanding (SUN) database [511, 512] fills the gap of the existing dataset with the limited scope of categories. This database has 899 categories and 130, 519 images, and only images with more than $2 0 0 \times 2 0 0$ pixels were kept. SUN397 is a more well-

sampled subset that maintains 397 categories with at least 100 images per category, in which other categories containing relatively few unique photographs are discarded. 

Places205 Places205 [513] dataset is another large scale scene dataset consists of 2, 448, 873 images from 205 scene categories. 

Cars [514] The dataset in the domain of cars contains 16, 185 color images of 196 classes (at the level of Make, Model, Year) of cars. For convenience, this dataset is split into training and testing sets in roughly equal quantities. 

Aircraft [515] It is another fine-grained visual classification designed for aircraft (also known as FGVC-Aircraft). A popular form of this dataset is the fine-grained recognition challenge 2013 (FGComp2013) [516] ran in parallel with the ILSVRC2013. There exist four-level hierarchies: Model, Variant, Family, Manufacturer, from finer to coarser to organize this database. The more detailed information is shown in [517]. 

Pets [518] It represents The Oxford-IIIT Pet Dataset that collects 37 pet categories with roughly 200 images per category. All images have an associated ground truth annotation of breed for classification, head ROI for detection, and pixel-level trimap for segmentation. 

Flowers [519] Similarly, Flowers is another domain dataset in flowers also collected by Oxford; it contains Oxford-17 Flowers of 17 categories and Oxford-102 Flowers of 102 categories. 

Describable Textures Dataset (DTD) [520] This is an evolving collection of textural images in the wild, which consists of 5, 640 images of 47 categories, with 120 images per category. 

iNaturalist2018 [521] It is a large-scale species classification competition conducted on the FGVC5 workshop at CVPR2018. This dataset contains over 8,000 species categories, with more than 450, 000 images in the training and validation dataset collected from iNaturalist [522]. 

JFT-300M [523] JFT-300M is an internal Google dataset introduced by Sun et al [523] and well-known from ViT Model [40]. It is labeled by algorithms that utilize human-computer communications and target classification tasks. This dataset finally contains 300M images with over 1000M labels, thus leading to the multiple labels attached to this large-scale dataset. 

Detection The detection is a popular task in the CV, and almost all the research is conducted on COCO and PASCAL VOC datasets. 

COCO [311] This is a large-scale dataset for object detection, segmentation, and caption; it contains 330, 000 RGB images, with more than 200, 000 labeled. There are 1.5 million object instances of 80 object categories involved. Thus, it is one of the most popular benchmark dataset in detection and segmentation in parallel with the following PASCAL VOC. 

PASCAL VOC [524] From 2005 through 2012, the dataset has run challenges assessing performance on object class recognition and has provided standardized image datasets for object class recognition. The main datasets used in self-supervised learning are VOC07, VOC11, and VOC12. Main competitions in VOC07 [525] contain classification and detection tasks; both of them consist of 20 objects and contain at least one object in each image. Thus, it is common to use VOC07 to serve as the downstream task for the detection. 

Segmentation The segmentation is a semantics-based pixel-level classification. These datasets are difficult to obtain and annotate, thus they are always used as a downstream task. 

VOC11 [526] & VOC12 [527] Both VOC11 and VOC12 contains classification, detection, and segmentation tasks in the main competition, thus leading to the common use of downstream task for the segmentation. 

ADE20K [528, 529] It collects 27, 574 images from both the SUN and Places205 databases, in which 25, 574 for training and 2, 000 for testing. All 707, 868 objects from 3, 688 categories existing in images are annotated. Especially, this dataset contains 193, 238 annotated object parts and parts of parts, and additional attributes, annotation time, depth ordering for the benefit of the research community. 

NYU-Depth V2 [530] This is a dataset consisting of images and video sequences from 464 indoor scenes that are recorded by both the RGB and Depth cameras from 3 cities. It contains 1, 449 images with the ground truth of depth, and the original RGB values are also provided. In addition, there are 407, 024 new unlabeled frames and additional class labels for the objects in images. 

Cityscapes [531, 532] It is a dataset of urban street scenes from 50 cities with the ground truth of semantic segmentation. The main instances are vehicles, people, and construction. The high-quality dense pixel annotations contain a volume of 5, 000 images. In addition to the fine annotations, coarser polygonal annotations are provided for a set of 20, 000 images. Moreover, the videos consist of not consistent images with high-quality annotations, and these annotated images with consistently changing views are provided for researchers. 

LVIS [533] It is a dataset for large vocabulary instance segmentation. It features that 1) a category or word in one image is related to the only segmentation object; 2) more than 1, 200 categories are extracted from roughly 160, 000 images; 3) long tails phenomenon exist in these categories; and 4) more than 2, 000, 000 high-quality instance segmentation masks. 

Densely Annotated VIdeo Segmentation (DAVIS) [534] It is a video dataset designed for the in-depth analysis of the SOTA in video object segmentation, in which DAVIS 2017 [535] contains both semisupervised (human-guided at the testing time) and unsupervised (human non-guided at test time) video sequences with multiple annotated instances. 

Others There are many datasets designed for special visual tasks such as inpainting. In addition, this part covers the data collection in the wild. 

Paris StreetView [536] The dataset is designed for image inpainting task, which contains 14, 900 training images and 100 testing images collected from street views of Paris. This dataset is collected from Google Street View and mainly focuses on the buildings in the city. 

Moving-MNIST [537] Based on MNIST, it is a video dataset designed for evaluating sequence prediction or reconstruction, which contains 10, 000 sequences. Each video is long of 20 frames and consisted of two digits (possibly overlapped) moving inside a $6 4 \times 6 4$ patch. The first benchmark is reported on [538] by the method of LSTMs. 

Yahoo Flickr Creative Commons 100 Million (YFCC100M) [539, 540] The dataset is the largest public multimedia collection that is allowed to search by users for their own targets; this dataset can browse both images and videos. It is free and for researchers to explore and investigate subsets of the YFCC100M in real time. Subsets of the complete dataset can be retrieved by any keyword search and reviewed directly. In addition, the text information attached to any image or video is abundant, such as containing location information and user tags. Briefly, it is more a multimedia library than a domain dataset. 

Data in the Wild More generalized dataset concept in the self-supervised learning era is composed of 

multimedia websites, APP, or search engines such as Instagram, Flickr, Google Images, etc. I think pictures in the wild will play a major role in the future study of CV because of the quantity of data, the computation source, and the learning power of PFM. 

# G.3 Downstream Tasks and Datasets on Graph

The purpose of the pretraining graph model is to improve the performance of downstream tasks. According to the different analysis objects of the downstream tasks, they can be divided into nodes, edges, and graphs. Meanwhile, the PFMs of GL have been widely used in a mass of fields. In this section, we combine the downstream tasks to conduct statistics on the pretraining datasets and the downstream task datasets. 

Node-Level Tasks Nodes are the most basic element of the graph, so lots of downstream tasks mainly focus on the analysis of nodes. 

Node Classification Node ClassiFication (NCF) is one of the most prevalent graph-based tasks, which has important analytical value in most of the different types of graph data. Different from the pseudo-labels assigned to nodes in the graph in self-supervised methods, the labels in NCF often come from external information such as manual annotation. Based on Definition 7 and 8, NCF can be divided into two types: transductive and inductive according to the visibility during training, verification, and testing. In addition, the result of NCF can be single-label or multi-label according to the mutual exclusion of labels. The statistical results of common NFC datasets are shown in Table 7. 

Node Clustering The goal of Node ClusterIng (NCI) is to divide a graph into different classes or clusters according to a certain standard so that the correlation of nodes in the same cluster is as large as possible, and the irrelevance of nodes that are not in the same cluster is also minimized. Although in the above-mentioned pretraining tasks, NCI is used as a pretext task has appeared, NCI can still test pretraining graph models based on other pretext tasks. 

Top-K Search The goal of task Top-K Search (TKS) is to search the K nodes with the highest predefined associations for a given node in the graph. Usually, TKS is used for search tasks such as recommendation and alignment. The detailed statistical results of the datasets are shown in Table 7. 

Link-Level Tasks The edge is also an important part of the graph structure, which associates independent nodes and is the key to distinguishing graph data from non-relational data. Especially in some specific fields (e.g., molecules, proteins), edges contain real information, so there are various tasks related to edges. 

Link Classification Similar to the NCF, the Link Classification (LC) also assigns one or more labels to a given edge. In fact, in LC, the nodes at both ends of the edge are still taken into consideration. 

Link Prediction Link Prediction (LP) is a common graph task (e.g., knowledge graph). The goal of LP is to predict edges that are removed or may exist in the graph. Similar to NCI, LP is also one of the pretext tasks in self-supervised learning, and its statistic results as shown in Table 8. 

Top-K Recommendation Top-K Recommendation (TKR) is exactly the same as the definition of TKS, the difference lies in the sorting goal. 

Graph-Level Tasks The graph-level task generally focuses on the distribution of nodes, edges, and attributes in a given graph, in order to infer the possible properties of the entire graph. 


Table 7: The statistics of the datasets for node-level tasks. Homogeneous:Hom, Heterogeneous:Het.


<table><tr><td>Task</td><td>Name</td><td>Usage</td><td>Source</td><td>Type</td><td>Nodes</td><td>Edges</td><td>Class</td><td>Features</td><td>Related Paper</td></tr><tr><td rowspan="36">NCF</td><td>Academia</td><td>pretrain</td><td>Citation</td><td>Hom</td><td>138K</td><td>739K</td><td>-</td><td>-</td><td>[195]</td></tr><tr><td>DBLP (SNAP)</td><td>pretrain</td><td>Citation</td><td>Hom</td><td>317K</td><td>2M</td><td>-</td><td>-</td><td>[195]</td></tr><tr><td>DBLP (NetRep)</td><td>pretrain</td><td>Citation</td><td>Hom</td><td>540K</td><td>30M</td><td>-</td><td>-</td><td>[195]</td></tr><tr><td>IMDB</td><td>pretrain</td><td>Movie</td><td>Hom</td><td>896K</td><td>8M</td><td>-</td><td>-</td><td>[195]</td></tr><tr><td>Facebook</td><td>pretrain</td><td>Social</td><td>Hom</td><td>3M</td><td>47M</td><td>-</td><td>-</td><td>[195]</td></tr><tr><td>LiveJournal</td><td>pretrain</td><td>Social</td><td>Hom</td><td>4M</td><td>86M</td><td>-</td><td>-</td><td>[195]</td></tr><tr><td>Cora</td><td>Downstream</td><td>Citation</td><td>Hom</td><td>2,708</td><td>5,429</td><td>7</td><td>1,433</td><td>[203, 188, 214, 194, 224] [200, 201, 202, 198, 209, 211]</td></tr><tr><td>CiteSeer</td><td>Downstream</td><td>Citation</td><td>Hom</td><td>3,327</td><td>4,732</td><td>6</td><td>3,703</td><td>[203, 188, 194, 224] [201, 202, 198, 209, 211]</td></tr><tr><td>PubMed</td><td>Downstream</td><td>Citation</td><td>Hom</td><td>19K</td><td>44K</td><td>3</td><td>500</td><td>[203, 201, 202, 198, 188] [209, 211, 194, 224, 200]</td></tr><tr><td>ACM</td><td>Downstream</td><td>Citation</td><td>Hom</td><td>8,994</td><td>26K</td><td>4</td><td>1,902</td><td>[215]</td></tr><tr><td>Cora-Full</td><td>Downstream</td><td>Citation</td><td>Hom</td><td>20K</td><td>63K</td><td>70</td><td>500</td><td>[224, 541]</td></tr><tr><td>Cora-ML</td><td>Downstream</td><td>Citation</td><td>Hom</td><td>2,995</td><td>8,158</td><td>7</td><td>2879</td><td>[224]</td></tr><tr><td>Reddit-233K</td><td>Downstream</td><td>Social</td><td>Hom</td><td>233K</td><td>57M</td><td>210</td><td>5,414</td><td>[189, 214, 201, 202]</td></tr><tr><td>BlogCatalog</td><td>Downstream</td><td>Social</td><td>Hom</td><td>10K</td><td>334K</td><td>39</td><td>-</td><td>[191, 192]</td></tr><tr><td>YouTube</td><td>Downstream</td><td>Social</td><td>Hom</td><td>1M</td><td>3M</td><td>47</td><td>-</td><td>[191]</td></tr><tr><td>Reddit-231K</td><td>Downstream</td><td>Social</td><td>Hom</td><td>231K</td><td>11M</td><td>41</td><td>602</td><td>[542, 543, 200, 211]</td></tr><tr><td>Amazon</td><td>Downstream</td><td>Social</td><td>Het</td><td>130M</td><td>-</td><td>-</td><td>-</td><td>[212]</td></tr><tr><td>PPI-30K</td><td>Downstream</td><td>Protein</td><td>Het</td><td>3,890</td><td>77K</td><td>50</td><td>-</td><td>[192, 200]</td></tr><tr><td>PPI-57K</td><td>Downstream</td><td>Protein</td><td>Het</td><td>57K</td><td>819K</td><td>121</td><td>50</td><td>[542, 224, 543, 202, 211]</td></tr><tr><td>IMDB</td><td>Downstream</td><td>Movie</td><td>Hom</td><td>12K</td><td>37K</td><td>4</td><td>1,256</td><td>[215]</td></tr><tr><td>Four-Univ</td><td>Downstream</td><td>Movie</td><td>Hom</td><td>4,518</td><td>3,426</td><td>6</td><td>2,000</td><td>[224]</td></tr><tr><td>Chameleon</td><td>Downstream</td><td>Web</td><td>Hom</td><td>2,277</td><td>36K</td><td>6</td><td>500</td><td>[224]</td></tr><tr><td>Crocodile</td><td>Downstream</td><td>Web</td><td>Hom</td><td>12K</td><td>180K</td><td>6</td><td>500</td><td>[224]</td></tr><tr><td>Flickr-89K</td><td>Downstream</td><td>Web</td><td>Hom</td><td>89K</td><td>450K</td><td>7</td><td>500</td><td>[224, 202]</td></tr><tr><td>ogbn-arxiv</td><td>Downstream</td><td>Web</td><td>Hom</td><td>169K</td><td>117K</td><td>40</td><td>128</td><td>[224]</td></tr><tr><td>Wiki-CS</td><td>Downstream</td><td>Web</td><td>Hom</td><td>12K</td><td>277K</td><td>10</td><td>300</td><td>[224, 541]</td></tr><tr><td>DBLP</td><td>Downstream</td><td>Web</td><td>Hom</td><td>17K</td><td>53K</td><td>4</td><td>1639</td><td>[224, 543]</td></tr><tr><td>Computers</td><td>Downstream</td><td>Co-purchase</td><td>Hom</td><td>14K</td><td>246K</td><td>10</td><td>767</td><td>[224, 198, 209, 541]</td></tr><tr><td>Photo</td><td>Downstream</td><td>Co-purchase</td><td>Hom</td><td>7,650</td><td>119K</td><td>8</td><td>745</td><td>[224, 198, 209, 541, 544]</td></tr><tr><td>CS</td><td>Downstream</td><td>Co-author</td><td>Hom</td><td>18K</td><td>82K</td><td>15</td><td>500</td><td>[224, 198, 209, 541, 544]</td></tr><tr><td>Physics</td><td>Downstream</td><td>Co-author</td><td>Hom</td><td>35K</td><td>248K</td><td>5</td><td>500</td><td>[224, 198, 541]</td></tr><tr><td>H-index</td><td>Downstream</td><td>Co-author</td><td>Hom</td><td>5,000</td><td>44K</td><td>-</td><td>-</td><td>[195]</td></tr><tr><td>Flickr-81K</td><td>Downstream</td><td>Photo</td><td>Hom</td><td>81K</td><td>6M</td><td>195</td><td>-</td><td>[191]</td></tr><tr><td>Wikipedia</td><td>Downstream</td><td>Word</td><td>Hom</td><td>4,777</td><td>185K</td><td>40</td><td>-</td><td>[192]</td></tr><tr><td>US-Airport</td><td>Downstream</td><td>Airline</td><td>Hom</td><td>1,190</td><td>13K</td><td>-</td><td>-</td><td>[195]</td></tr><tr><td>OAG</td><td>Downstream</td><td>Academic</td><td>Het</td><td>178M</td><td>2B</td><td>-</td><td>-</td><td>[212]</td></tr><tr><td rowspan="3">NTKS</td><td>KDD-ICDM</td><td>Downstream</td><td>Co-author</td><td>Hom</td><td>2,867/2,607</td><td>7,637/4,774</td><td>697</td><td>-</td><td>[195]</td></tr><tr><td>SIGIR-CIKM</td><td>Downstream</td><td>Co-author</td><td>Hom</td><td>2,851/3,548</td><td>6,354/7,076</td><td>874</td><td>-</td><td>[195]</td></tr><tr><td>SIGMOD-ICDE</td><td>Downstream</td><td>Co-author</td><td>Hom</td><td>2,626/2,559</td><td>8,304/6,668</td><td>898</td><td>-</td><td>[195]</td></tr></table>


Table 8: The statistics of the datasets for LC. Homogeneous:Hom, Heterogeneous:Het.


<table><tr><td>Name</td><td>Usage</td><td>Source</td><td>Type</td><td>Nodes</td><td>Edges</td><td>Class</td><td>Features</td><td>Related Paper</td></tr><tr><td>Cora</td><td>Downstream</td><td>Citation</td><td>Hom</td><td>2,708</td><td>5,429</td><td>7</td><td>1,433</td><td>[203, 188, 189, 545, 546, 214, 194, 224, 543, 542] [200, 201, 202, 198, 209, 210, 211, 544]</td></tr><tr><td>CiteSeer</td><td>Downstream</td><td>Citation</td><td>Hom</td><td>3,327</td><td>4,732</td><td>6</td><td>3,703</td><td>[203, 188, 189, 546, 542, 194, 224, 200, 543] [201, 202, 198, 209, 210, 211, 541, 544]</td></tr><tr><td>PubMed</td><td>Downstream</td><td>Citation</td><td>Hom</td><td>19K</td><td>44K</td><td>3</td><td>500</td><td>[203, 188, 189, 545, 546, 542, 194, 224, 543, 200] [201, 202, 198, 209, 210, 211, 544]</td></tr><tr><td>ML-100K</td><td>Downstream</td><td>Movie</td><td>Hom</td><td>2,625</td><td>100K</td><td>5</td><td>-</td><td>[545]</td></tr><tr><td>ML-1M</td><td>Downstream</td><td>Movie</td><td>Hom</td><td>9,940</td><td>1M</td><td>5</td><td>-</td><td>[545]</td></tr><tr><td>BlogCatalog-5K</td><td>Downstream</td><td>Social</td><td>Hom</td><td>5,196</td><td>172K</td><td>6</td><td>8,189</td><td>[542, 211]</td></tr><tr><td>Amazon</td><td>Downstream</td><td>Social</td><td>Het</td><td>130M</td><td>-</td><td>-</td><td>-</td><td>[212]</td></tr><tr><td>PPI-57K</td><td>Downstream</td><td>Protein</td><td>Het</td><td>57K</td><td>819K</td><td>121</td><td>50</td><td>[542, 224, 543, 202, 211]</td></tr><tr><td>Flickr-7K</td><td>Downstream</td><td>Photo</td><td>Hom</td><td>7,575</td><td>240M</td><td>9</td><td>12,047</td><td>[542, 211]</td></tr><tr><td>Last-FM</td><td>Downstream</td><td>Music</td><td>Hom</td><td>15K</td><td>73K</td><td>122</td><td>-</td><td>[215]</td></tr><tr><td>Book-Crossing</td><td>Downstream</td><td>Book</td><td>Hom</td><td>111K</td><td>443K</td><td>52</td><td>-</td><td>[215]</td></tr><tr><td>OAG</td><td>Downstream</td><td>Academic</td><td>Het</td><td>178M</td><td>2B</td><td>-</td><td>-</td><td>[212]</td></tr></table>

Graph Classification Graph Classification (GC) is commonly used in social, molecular, and protein graph data, which aims to predict the property of the given community, chemical compound, and protein. The statistic results as shown in Table 9. 


Table 9: The statistics of the datasets for GC. Homogeneous:Hom, Heterogeneous:Het.


<table><tr><td>Name</td><td>Usage</td><td>Source</td><td>Type</td><td>Graphs</td><td>Nodes</td><td>Edges</td><td>Class</td><td>Related Paper</td></tr><tr><td>ZINC15</td><td>Pretraining</td><td>Molecule</td><td>Hom</td><td>2M</td><td>-</td><td>-</td><td>-</td><td>[190, 204]</td></tr><tr><td>ChEMBL</td><td>Pretraining</td><td>Molecule</td><td>Hom</td><td>456K</td><td>-</td><td>-</td><td>-</td><td>[190, 204]</td></tr><tr><td>PPI-pre</td><td>Pretraining</td><td>Protein</td><td>Het</td><td>395K</td><td>-</td><td>-</td><td>-</td><td>[190]</td></tr><tr><td>MUTAG</td><td>Downstream</td><td>Molecule</td><td>Hom</td><td>188</td><td>-</td><td>-</td><td>2</td><td>[190, 547, 201, 216, 199, 218, 225, 548]</td></tr><tr><td>PTC</td><td>Downstream</td><td>Molecule</td><td>Hom</td><td>344</td><td>-</td><td>-</td><td>2</td><td>[190, 547, 201, 216, 199, 548]</td></tr><tr><td>BBBP</td><td>Downstream</td><td>Molecule</td><td>Hom</td><td>2,039</td><td>-</td><td>-</td><td>2</td><td>[190, 204, 549, 218, 220, 225]</td></tr><tr><td>Tox21</td><td>Downstream</td><td>Molecule</td><td>Hom</td><td>7,831</td><td>-</td><td>-</td><td>24</td><td>[190, 204, 549, 218, 220, 225]</td></tr><tr><td>ToxCast</td><td>Downstream</td><td>Molecule</td><td>Hom</td><td>8,575</td><td>-</td><td>-</td><td>1,234</td><td>[190, 204, 549, 218, 220, 225]</td></tr><tr><td>SIDER</td><td>Downstream</td><td>Molecule</td><td>Hom</td><td>1,427</td><td>-</td><td>-</td><td>54</td><td>[190, 204, 549, 218, 220, 225]</td></tr><tr><td>ClinTox</td><td>Downstream</td><td>Molecule</td><td>Hom</td><td>1,478</td><td>-</td><td>-</td><td>4</td><td>[190, 204, 549, 218, 220, 225]</td></tr><tr><td>MUV</td><td>Downstream</td><td>Molecule</td><td>Hom</td><td>93K</td><td>-</td><td>-</td><td>34</td><td>[190, 218, 220]</td></tr><tr><td>HIV</td><td>Downstream</td><td>Molecule</td><td>Hom</td><td>41K</td><td>-</td><td>-</td><td>2</td><td>[190, 549, 218, 220]</td></tr><tr><td>BACE</td><td>Downstream</td><td>Molecule</td><td>Hom</td><td>1,513</td><td>-</td><td>-</td><td>2</td><td>[190, 549, 218, 220, 225]</td></tr><tr><td>PPI-88K</td><td>Downstream</td><td>Protein</td><td>Het</td><td>88K</td><td>-</td><td>-</td><td>80</td><td>[190]</td></tr><tr><td>IMDB-M</td><td>Downstream</td><td>Movie</td><td>Hom</td><td>1,500</td><td>19K</td><td>99K</td><td>3</td><td>[545, 195, 547, 201, 216]</td></tr><tr><td>IMDB-B</td><td>Downstream</td><td>Movie</td><td>Hom</td><td>1,000</td><td>19K</td><td>97K</td><td>2</td><td>[545, 195, 547, 201, 216, 218]</td></tr><tr><td>FreeSolv</td><td>Downstream</td><td>Molecule</td><td>Hom</td><td>642</td><td>-</td><td>-</td><td>-</td><td>[204]</td></tr><tr><td>ESOL</td><td>Downstream</td><td>Molecule</td><td>Hom</td><td>1,128</td><td>-</td><td>-</td><td>-</td><td>[204]</td></tr><tr><td>Lipophilicity</td><td>Downstream</td><td>Molecule</td><td>Hom</td><td>4,200</td><td>-</td><td>-</td><td>-</td><td>[204]</td></tr><tr><td>QM7</td><td>Downstream</td><td>Molecule</td><td>Hom</td><td>6,830</td><td>-</td><td>-</td><td>-</td><td>[204]</td></tr><tr><td>QM8</td><td>Downstream</td><td>Molecule</td><td>Hom</td><td>22K</td><td>-</td><td>-</td><td>-</td><td>[204]</td></tr><tr><td>COLLAB</td><td>Downstream</td><td>Co-author</td><td>Hom</td><td>5,000</td><td>373K</td><td>-</td><td>3</td><td>[195, 547, 218, 548]</td></tr><tr><td>RDT-B</td><td>Downstream</td><td>Co-author</td><td>Hom</td><td>2,000</td><td>859K</td><td>-</td><td>2</td><td>[195, 216, 218, 548]</td></tr><tr><td>RDT-M</td><td>Downstream</td><td>Co-author</td><td>Hom</td><td>5,000</td><td>3M</td><td>-</td><td>5</td><td>[195, 216, 218, 548]</td></tr><tr><td>NCI1</td><td>Downstream</td><td>Molecule</td><td>Hom</td><td>4,110</td><td>123K</td><td>132K</td><td>2</td><td>[197, 219, 547, 199, 218, 548]</td></tr><tr><td>NCI109</td><td>Downstream</td><td>Molecule</td><td>Hom</td><td>4,127</td><td>123K</td><td>133K</td><td>2</td><td>[199]</td></tr><tr><td>PROTEINS</td><td>Downstream</td><td>Molecule</td><td>Hom</td><td>1,113</td><td>44K</td><td>81K</td><td>2</td><td>[197, 219, 199, 218, 548]</td></tr><tr><td>D&amp;D</td><td>Downstream</td><td>Molecule</td><td>Hom</td><td>1,178</td><td>335K</td><td>843K</td><td>2</td><td>[199, 218]</td></tr><tr><td>Mutagenicity</td><td>Downstream</td><td>Molecule</td><td>Hom</td><td>4,337</td><td>131K</td><td>134K</td><td>2</td><td>[219]</td></tr><tr><td>METR-LA</td><td>Downstream</td><td>Traffic</td><td>Hom</td><td>1</td><td>207</td><td>-</td><td>-</td><td>[550]</td></tr></table>

Data Source The PFMs of GL have been widely used in a mass of fields. We will descript the details of the pretraining datasets and the downstream task datasets. 

Citation and Co-author network A citation is a basic local representation, whose structure reflects the citation relationships of papers in a research direction or field. Specifically, a citation network is a kind of relational data composed of research papers as nodes and citation relations as edges. Among them, the citation network used in the GL model usually comes from local samples of common citation databases, e.g., Cora, Citeseer, and PubMed, and serves as downstream tasks. Similarly, the co-author network is a dataset of scientific collaboration that corresponds to a researcher’s ego network, in which the researcher and their collaborators are nodes and an edge indicates collaboration between two researchers. According to different requirements of downstream tasks, such co-author networks can be used for various tasks, e.g., node classification and graph classification. 

Molecular and protein network A molecular network usually refers to a compound composed of atoms and atomic bonds, and predicting the properties of the compound is usually regarded as a graph classification task. For example, MUTAG is a collection of nitroaromatic compounds whose goal is to predict their mutagenicity to Salmonella typhimurium. PTC uses a graph to show the structure of multiple compounds and aims to predict the carcinogenicity of different compounds in rats. The protein network is a collection of proteins classified as either enzymes or non-enzymes. The amino acids are represented by nodes, and two nodes are connected by an edge if they are less than 6 Angstroms apart. 

Social and Movie network The social network is the social-relational data in the real network environment, which usually represents the relationship between users or posts. For instance, Reddit is a graph dataset comprised of Reddit posts made in September 2014. BlogCatalog is a graph dataset that represents a network of social relationships between bloggers who are listed on the BlogCatalog website. The movie network is usually composed of actors and their co-occurrence participation in the movie. For example, IMDB-B is a movie collaboration dataset that contains a large number of self-networks of actors who play movie roles in IMDB. Nodes in each graph represent actors/actresses, and if they appear in the same film, an edge connects them. These graphs are based on action and romance genres. The difference between IMDB-M and IMDB-B is that a node in the graph represents one or more actors. 

Others Some of the rarer graph data are used to test the universality of the PFM, such as word networks (Wikipedia), book networks (Book-crossing), and airline networks (US-Airport). In addition, there are also some special graph structures adapted to specific models, such as spatiotemporal graphs (METR-LA).

[1] R. Bommasani, D. A. Hudson, E. Adeli, R. Altman, S. Arora, S. von Arx, M. S. Bernstein, J. Bohg, A. Bosselut, E. Brunskill, et al., “On the opportunities and risks of foundation models,” arXiv preprint arXiv:2108.07258, 2021. 





[2] G. G. Chowdhury, “Natural language processing,” Annual review of information science and technology, 2003. 





[3] D. Forsyth and J. Ponce, Computer vision: A modern approach. 2011. 





[4] J. A. Bondy, U. S. R. Murty, et al., Graph theory with applications. 1976. 





[5] X. Qiu, T. Sun, Y. Xu, Y. Shao, N. Dai, and X. Huang, “Pre-trained models for natural language processing: A survey,” Science China Technological Sciences, 2020. 





[6] J. Li, T. Tang, W. X. Zhao, and J.-R. Wen, “Pretrained language models for text generation: A survey,” arXiv, 2021. 





[7] K. Han, Y. Wang, H. Chen, X. Chen, J. Guo, Z. Liu, Y. Tang, A. Xiao, C. Xu, Y. Xu, et al., “A survey on visual transformer,” arXiv, 2020. 





[8] S. Sanchez, H. Romero, and A. Morales, “A review: Comparison of performance metrics of pretrained models for object detection using the tensorflow framework,” in IOP Conference Series: Materials Science and Engineering. 





[9] W. Hu, B. Liu, J. Gomes, M. Zitnik, P. Liang, V. Pande, and J. Leskovec, “Pre-training graph neural networks,” arXiv, 2019. 





[10] F. Zhuang, Z. Qi, K. Duan, D. Xi, Y. Zhu, H. Zhu, H. Xiong, and Q. He, “A comprehensive survey on transfer learning,” Proceedings of the IEEE, 2020. 





[11] Y. Bengio, R. Ducharme, P. Vincent, and C. Janvin, “A neural probabilistic language model,” J. Mach. Learn. Res., 2003. 





[12] T. Mikolov, K. Chen, G. Corrado, and J. Dean, “Efficient estimation of word representations in vector space,” in Proc. ICLR, 2013, 2013. 





[13] J. Devlin, M. Chang, K. Lee, and K. Toutanova, “BERT: pre-training of deep bidirectional transformers for language understanding,” in NAACL-HLT, 2019. 





[14] Z. Yang, Z. Dai, Y. Yang, J. G. Carbonell, R. Salakhutdinov, and Q. V. Le, “Xlnet: Generalized autoregressive pretraining for language understanding,” in NeurIPS. 





[15] M. Chen, J. Tworek, H. Jun, Q. Yuan, H. P. d. O. Pinto, J. Kaplan, H. Edwards, Y. Burda, N. Joseph, G. Brockman, et al., “Evaluating large language models trained on code,” arXiv preprint arXiv:2107.03374, 2021. 





[16] A. Neelakantan, T. Xu, R. Puri, A. Radford, J. M. Han, J. Tworek, Q. Yuan, N. Tezak, J. W. Kim, C. Hallacy, et al., “Text and code embeddings by contrastive pre-training,” arXiv preprint arXiv:2201.10005, 2022. 





[17] P. F. Christiano, J. Leike, T. Brown, M. Martic, S. Legg, and D. Amodei, “Deep reinforcement learning from human preferences,” Advances in neural information processing systems, vol. 30, 2017. 





[18] N. Stiennon, L. Ouyang, J. Wu, D. Ziegler, R. Lowe, C. Voss, A. Radford, D. Amodei, and P. F. Christiano, “Learning to summarize with human feedback,” Advances in Neural Information Processing Systems, vol. 33, pp. 3008–3021, 2020. 





[19] L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. L. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, et al., “Training language models to follow instructions with human feedback,” arXiv preprint arXiv:2203.02155, 2022. 





[20] T. B. Brown, B. Mann, N. Ryder, M. Subbiah, J. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, et al., “Language models are few-shot learners,” arXiv, 2020. 





[21] B. Lester, R. Al-Rfou, and N. Constant, “The power of scale for parameter-efficient prompt tuning,” in Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pp. 3045–3059, 2021. 





[22] T. Schick and H. Schütze, “Exploiting cloze-questions for few-shot text classification and natural language inference,” in Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume, pp. 255–269, 2021. 





[23] Z. Zhang, A. Zhang, M. Li, and A. Smola, “Automatic chain of thought prompting in large language models,” in International Conference on Learning Representations, 2023. 





[24] J. Wei, X. Wang, D. Schuurmans, M. Bosma, F. Xia, E. H. Chi, Q. V. Le, D. Zhou, et al., “Chainof-thought prompting elicits reasoning in large language models,” in Advances in Neural Information Processing Systems, 2022. 





[25] OpenAI, “Gpt-4 technical report,” 2023. 





[26] P. Wang, A. Yang, R. Men, J. Lin, S. Bai, Z. Li, J. Ma, C. Zhou, J. Zhou, and H. Yang, “Unifying architectures, tasks, and modalities through a simple sequence-to-sequence learning framework,” arXiv preprint arXiv:2202.03052, 2022. 





[27] J. Lu, C. Clark, R. Zellers, R. Mottaghi, and A. Kembhavi, “Unified-io: A unified model for vision, language, and multi-modal tasks,” arXiv preprint arXiv:2206.08916, 2022. 





[28] A. Singh, R. Hu, V. Goswami, G. Couairon, W. Galuba, M. Rohrbach, and D. Kiela, “Flava: A foundational language and vision alignment model,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 15638–15650, 2022. 





[29] W. Wang, H. Bao, L. Dong, J. Bjorck, Z. Peng, Q. Liu, K. Aggarwal, O. K. Mohammed, S. Singhal, S. Som, et al., “Image as a foreign language: Beit pretraining for all vision and vision-language tasks,” arXiv preprint arXiv:2208.10442, 2022. 





[30] K. Clark, M. Luong, Q. V. Le, and C. D. Manning, “ELECTRA: pre-training text encoders as discriminators rather than generators,” in ICLR, 2020. 





[31] E. Wallace, P. Rodriguez, S. Feng, I. Yamada, and J. Boyd-Graber, “Trick me if you can: Human-inthe-loop generation of adversarial examples for question answering,” 2019. 





[32] Y. Nie, A. Williams, E. Dinan, M. Bansal, J. Weston, and D. Kiela, “Adversarial NLI: A new benchmark for natural language understanding,” in ACL. 





[33] T. Niven and H. Kao, “Probing neural network comprehension of natural language arguments,” in ACL. 





[34] G. Wang, N. Ivanov, B. Chen, Q. Wang, and Q. Yan, “Graph learning for interactive threat detection in heterogeneous smart home rule data,” in 2023 ACM SIGMOD International Conference on Management of Data, ACM, 2023. 





[35] M. A. Gordon, K. Duh, and N. Andrews, “Compressing BERT: studying the effects of weight pruning on transfer learning,” in RepL4NLP@ACL. 





[36] Z. Lan, M. Chen, S. Goodman, K. Gimpel, P. Sharma, and R. Soricut, “ALBERT: A lite BERT for self-supervised learning of language representations,” in ICLR, 2020. 





[37] X. Han, Z. Zhang, N. Ding, Y. Gu, X. Liu, Y. Huo, J. Qiu, L. Zhang, W. Han, M. Huang, et al., “Pre-trained models: Past, present and future,” AI Open, 2021. 





[38] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, “Attention is all you need,” arXiv, 2017. 





[39] M.-H. Guo, T.-X. Xu, J.-J. Liu, Z.-N. Liu, P.-T. Jiang, T.-J. Mu, S.-H. Zhang, R. R. Martin, M.-M. Cheng, and S.-M. Hu, “Attention mechanisms in computer vision: A survey,” Computational Visual Media, vol. 8, no. 3, pp. 331–368, 2022. 





[40] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, et al., “An image is worth 16x16 words: Transformers for image recognition at scale,” arXiv preprint arXiv:2010.11929, 2020. 





[41] S. Yun, M. Jeong, R. Kim, J. Kang, and H. J. Kim, “Graph transformer networks,” Advances in neural information processing systems, vol. 32, 2019. 





[42] B. M. P. P. J. H. J. G. A. S. M. C. R. G. I. A. R. J. L. B. M. T. A. A. X. W. C. R. M. M. J. P. U. E. M. K. S. v. S. G. F. E. A. M. F. Y. A. O. F. H. J. B. M. P. C. A. A. G. V. B. C. V. Y. T. T. M. A. K. F. P. D. T. T. K. M. L. X. Z. D. K. J. H. N. H. Mostafa Dehghani, Josip Djolonga, “Scaling vision transformers to 22 billion parameters,” arXiv preprint arXiv:2302.05442, 2023. 





[43] A. Chowdhery, S. Narang, J. Devlin, M. Bosma, G. Mishra, A. Roberts, P. Barham, H. W. Chung, C. Sutton, S. Gehrmann, et al., “Palm: Scaling language modeling with pathways,” arXiv preprint arXiv:2204.02311, 2022. 





[44] M. Joshi, D. Chen, Y. Liu, D. S. Weld, L. Zettlemoyer, and O. Levy, “Spanbert: Improving pretraining by representing and predicting spans,” Trans. Assoc. Comput. Linguistics, 2020. 





[45] M. Lewis, Y. Liu, N. Goyal, M. Ghazvininejad, A. Mohamed, O. Levy, V. Stoyanov, and L. Zettlemoyer, “BART: denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension,” in ACL. 





[46] K. Clark, M.-T. Luong, Q. V. Le, and C. D. Manning, “Electra: Pre-training text encoders as discriminators rather than generators,” arXiv, 2020. 





[47] Z. Lan, M. Chen, S. Goodman, K. Gimpel, P. Sharma, and R. Soricut, “Albert: A lite bert for selfsupervised learning of language representations,” arXiv, 2019. 





[48] J. Donahue, P. Krähenbühl, and T. Darrell, “Adversarial feature learning,” arXiv, 2016. 





[49] M. Caron, P. Bojanowski, A. Joulin, and M. Douze, “Deep clustering for unsupervised learning of visual features,” in ECCV. 





[50] A. Radford, K. Narasimhan, T. Salimans, and I. Sutskever, “Improving language understanding by generative pre-training,” 2018. 





[51] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, and I. Sutskever, “Language models are unsupervised multitask learners,” OpenAI blog, 2019. 





[52] R. Caruana, “Multitask learning,” Machine learning, 1997. 





[53] M. E. Peters, M. Neumann, M. Iyyer, M. Gardner, C. Clark, K. Lee, and L. Zettlemoyer, “Deep contextualized word representations,” arXiv, 2018. 





[54] Y. Wu, M. Schuster, Z. Chen, Q. V. Le, M. Norouzi, W. Macherey, M. Krikun, Y. Cao, Q. Gao, K. Macherey, et al., “Google’s neural machine translation system: Bridging the gap between human and machine translation,” arXiv, 2016. 





[55] Y. Liu, M. Ott, N. Goyal, J. Du, M. Joshi, D. Chen, O. Levy, M. Lewis, L. Zettlemoyer, and V. Stoyanov, “Roberta: A robustly optimized BERT pretraining approach,” CoRR, 2019. 





[56] R. Sennrich, B. Haddow, and A. Birch, “Neural machine translation of rare words with subword units,” arXiv, 2015. 





[57] K. Song, X. Tan, T. Qin, J. Lu, and T. Liu, “Mpnet: Masked and permuted pre-training for language understanding,” in NeurIPS, 2020. 





[58] Q. Li, H. Peng, J. Li, C. Xia, R. Yang, L. Sun, P. S. Yu, and L. He, “A survey on text classification: From traditional to deep learning,” ACM Transactions on Intelligent Systems and Technology (TIST), vol. 13, no. 2, pp. 1–41, 2022. 





[59] K. Song, X. Tan, T. Qin, J. Lu, and T.-Y. Liu, “Mass: Masked sequence to sequence pre-training for language generation,” arXiv, 2019. 





[60] L. Dong, N. Yang, W. Wang, F. Wei, X. Liu, Y. Wang, J. Gao, M. Zhou, and H.-W. Hon, “Unified language model pre-training for natural language understanding and generation,” arXiv, 2019. 





[61] Y. Sun, S. Wang, Y. Li, S. Feng, X. Chen, H. Zhang, X. Tian, D. Zhu, H. Tian, and H. Wu, “Ernie: Enhanced representation through knowledge integration,” arXiv, 2019. 





[62] Y. Sun, S. Wang, Y. Li, S. Feng, H. Tian, H. Wu, and H. Wang, “Ernie 2.0: A continual pre-training framework for language understanding,” in AAAI. 





[63] Y. Cui, W. Che, T. Liu, B. Qin, and Z. Yang, “Pre-training with whole word masking for chinese BERT,” T-ASL, 2021. 





[64] S. Diao, J. Bai, Y. Song, T. Zhang, and Y. Wang, “ZEN: pre-training chinese text encoder enhanced by n-gram representations,” in EMNLP. 





[65] H. Tsai, J. Riesa, M. Johnson, N. Arivazhagan, X. Li, and A. Archer, “Small and practical bert models for sequence labeling,” arXiv, 2019. 





[66] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal policy optimization algorithms,” arXiv preprint arXiv:1707.06347, 2017. 





[67] R. Thoppilan, D. De Freitas, J. Hall, N. Shazeer, A. Kulshreshtha, H.-T. Cheng, A. Jin, T. Bos, L. Baker, Y. Du, et al., “Lamda: Language models for dialog applications,” arXiv preprint arXiv:2201.08239, 2022. 





[68] T. Mikolov, I. Sutskever, K. Chen, G. Corrado, and J. Dean, “Distributed representations of words and phrases and their compositionality,” arXiv, 2013. 





[69] J. Pennington, R. Socher, and C. D. Manning, “Glove: Global vectors for word representation,” in EMNLP. 





[70] A. M. Dai and Q. V. Le, “Semi-supervised sequence learning,” arXiv, 2015. 





[71] P. Liu, X. Qiu, and X. Huang, “Recurrent neural network for text classification with multi-task learning,” arXiv, 2016. 





[72] P. Bojanowski, E. Grave, A. Joulin, and T. Mikolov, “Enriching word vectors with subword information,” TACL, 2017. 





[73] B. McCann, J. Bradbury, C. Xiong, and R. Socher, “Learned in translation: Contextualized word vectors,” arXiv, 2017. 





[74] Z. Dai, Z. Yang, Y. Yang, J. Carbonell, Q. V. Le, and R. Salakhutdinov, “Transformer-xl: Attentive language models beyond a fixed-length context,” arXiv, 2019. 





[75] L. Kong, C. d. M. d’Autume, W. Ling, L. Yu, Z. Dai, and D. Yogatama, “A mutual information maximization perspective of language representation learning,” arXiv, 2019. 





[76] W. Wang, B. Bi, M. Yan, C. Wu, Z. Bao, J. Xia, L. Peng, and L. Si, “Structbert: Incorporating language structures into pre-training for deep language understanding,” arXiv, 2019. 





[77] W. Xiong, J. Du, W. Y. Wang, and V. Stoyanov, “Pretrained encyclopedia: Weakly supervised knowledge-pretrained language model,” arXiv, 2019. 





[78] M. E. Peters, M. Neumann, R. L. Logan IV, R. Schwartz, V. Joshi, S. Singh, and N. A. Smith, “Knowledge enhanced contextual word representations,” arXiv, 2019. 





[79] H. Huang, Y. Liang, N. Duan, M. Gong, L. Shou, D. Jiang, and M. Zhou, “Unicoder: A universal language encoder by pre-training with multiple cross-lingual tasks,” arXiv, 2019. 





[80] J. M. Eisenschlos, S. Ruder, P. Czapla, M. Kardas, S. Gugger, and J. Howard, “Multifit: Efficient multi-lingual language model fine-tuning,” arXiv, 2019. 





[81] I. Beltagy, K. Lo, and A. Cohan, “Scibert: A pretrained language model for scientific text,” arXiv, 2019. 





[82] S. Sun, Y. Cheng, Z. Gan, and J. Liu, “Patient knowledge distillation for bert model compression,” arXiv, 2019. 





[83] G. Lample and A. Conneau, “Cross-lingual language model pretraining,” arXiv, 2019. 





[84] O. Zafrir, G. Boudoukh, P. Izsak, and M. Wasserblat, “Q8BERT: quantized 8bit BERT,” in EMC2@NeurIPS. 





[85] V. Sanh, L. Debut, J. Chaumond, and T. Wolf, “Distilbert, a distilled version of bert: smaller, faster, cheaper and lighter,” arXiv, 2019. 





[86] W. Liu, P. Zhou, Z. Zhao, Z. Wang, H. Deng, and Q. Ju, “Fastbert: a self-distilling bert with adaptive inference time,” arXiv, 2020. 





[87] L. Martin, B. Müller, P. J. O. Suárez, Y. Dupont, L. Romary, É. de la Clergerie, D. Seddah, and B. Sagot, “Camembert: a tasty french language model,” in ACL. 





[88] A. Conneau, K. Khandelwal, N. Goyal, V. Chaudhary, G. Wenzek, F. Guzmán, E. Grave, M. Ott, L. Zettlemoyer, and V. Stoyanov, “Unsupervised cross-lingual representation learning at scale,” in ACL. 





[89] N. Kitaev, L. Kaiser, and A. Levskaya, “Reformer: The efficient transformer,” in ICLR, 2020. 





[90] S. Shen, Z. Dong, J. Ye, L. Ma, Z. Yao, A. Gholami, M. W. Mahoney, and K. Keutzer, “Q-bert: Hessian based ultra low precision quantization of bert,” in AAAI. 





[91] Z. Chi, L. Dong, F. Wei, W. Wang, X.-L. Mao, and H. Huang, “Cross-lingual natural language generation via pre-training,” in AAAI. 





[92] W. Liu, P. Zhou, Z. Zhao, Z. Wang, Q. Ju, H. Deng, and P. Wang, “K-bert: Enabling language representation with knowledge graph,” in AAAI. 





[93] Z. Jiang, W. Yu, D. Zhou, Y. Chen, J. Feng, and S. Yan, “Convbert: Improving BERT with span-based dynamic convolution,” in NeurIPS, 2020. 





[94] W. Wang, F. Wei, L. Dong, H. Bao, N. Yang, and M. Zhou, “Minilm: Deep self-attention distillation for task-agnostic compression of pre-trained transformers,” in NeurIPS, 2020. 





[95] Y. Liu, J. Gu, N. Goyal, X. Li, S. Edunov, M. Ghazvininejad, M. Lewis, and L. Zettlemoyer, “Multilingual denoising pre-training for neural machine translation,” Trans. Assoc. Comput. Linguistics, 2020. 





[96] T. Sun, Y. Shao, X. Qiu, Q. Guo, Y. Hu, X. Huang, and Z. Zhang, “Colake: Contextualized language and knowledge embedding,” in COLING. 





[97] H. Le, L. Vial, J. Frej, V. Segonne, M. Coavoux, B. Lecouteux, A. Allauzen, B. Crabbé, L. Besacier, and D. Schwab, “Flaubert: Unsupervised language model pre-training for french,” in LREC. 





[98] T. Shen, Y. Mao, P. He, G. Long, A. Trischler, and W. Chen, “Exploiting structured knowledge in text via graph-guided representation learning,” in EMNLP. 





[99] X. Jiao, Y. Yin, L. Shang, X. Jiang, X. Chen, L. Li, F. Wang, and Q. Liu, “Tinybert: Distilling BERT for natural language understanding,” in EMNLP. 





[100] P. Delobelle, T. Winters, and B. Berendt, “Robbert: a dutch roberta-based language model,” in EMNLP. 





[101] B. He, D. Zhou, J. Xiao, X. Jiang, Q. Liu, N. J. Yuan, and T. Xu, “Integrating graph contextualized knowledge into pre-trained language models,” in EMNLP. 





[102] C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena, Y. Zhou, W. Li, and P. J. Liu, “Exploring the limits of transfer learning with a unified text-to-text transformer,” J. Mach. Learn. Res., 2020. 





[103] T. Schick and H. Schütze, “Exploiting cloze-questions for few-shot text classification and natural language inference,” in EACL, 2021. 





[104] X. Wang, T. Gao, Z. Zhu, Z. Zhang, Z. Liu, J. Li, and J. Tang, “KEPLER: A unified model for knowledge embedding and pre-trained language representation,” Trans. Assoc. Comput. Linguistics, 2021. 





[105] T. Gao, X. Yao, and D. Chen, “Simcse: Simple contrastive learning of sentence embeddings,” CoRR, 2021. 





[106] N. Du, Y. Huang, A. M. Dai, S. Tong, D. Lepikhin, Y. Xu, M. Krikun, Y. Zhou, A. W. Yu, O. Firat, et al., “Glam: Efficient scaling of language models with mixture-of-experts,” in International Conference on Machine Learning, pp. 5547–5569, PMLR, 2022. 





[107] Z. Chi, S. Huang, L. Dong, S. Ma, S. Singhal, P. Bajaj, X. Song, and F. Wei, “Xlm-e: Cross-lingual language model pre-training via electra,” arXiv preprint arXiv:2106.16138, 2021. 





[108] V. Sanh, A. Webson, C. Raffel, S. H. Bach, L. Sutawika, Z. Alyafeai, A. Chaffin, A. Stiegler, T. L. Scao, A. Raja, et al., “Multitask prompted training enables zero-shot task generalization,” arXiv preprint arXiv:2110.08207, 2021. 





[109] J. W. Rae, S. Borgeaud, T. Cai, K. Millican, J. Hoffmann, F. Song, J. Aslanides, S. Henderson, R. Ring, S. Young, et al., “Scaling language models: Methods, analysis & insights from training gopher,” arXiv preprint arXiv:2112.11446, 2021. 





[110] S. Smith, M. Patwary, B. Norick, P. LeGresley, S. Rajbhandari, J. Casper, Z. Liu, S. Prabhumoye, G. Zerveas, V. Korthikanti, et al., “Using deepspeed and megatron to train megatron-turing nlg 530b, a large-scale generative language model,” arXiv preprint arXiv:2201.11990, 2022. 





[111] J. Hoffmann, S. Borgeaud, A. Mensch, E. Buchatskaya, T. Cai, E. Rutherford, D. d. L. Casas, L. A. Hendricks, J. Welbl, A. Clark, et al., “Training compute-optimal large language models,” arXiv preprint arXiv:2203.15556, 2022. 





[112] S. Zhang, S. Roller, N. Goyal, M. Artetxe, M. Chen, S. Chen, C. Dewan, M. Diab, X. Li, X. V. Lin, et al., “Opt: Open pre-trained transformer language models,” arXiv preprint arXiv:2205.01068, 2022. 





[113] J. Wei, M. Bosma, V. Zhao, K. Guu, A. W. Yu, B. Lester, N. Du, A. M. Dai, and Q. V. Le, “Finetuned language models are zero-shot learners,” in International Conference on Learning Representations, 2022. 





[114] O. Honovich, T. Scialom, O. Levy, and T. Schick, “Unnatural instructions: Tuning language models with (almost) no human labor,” arXiv preprint arXiv:2212.09689, 2022. 





[115] Y. Wang, S. Mishra, P. Alipoormolabashi, Y. Kordi, A. Mirzaei, A. Naik, A. Ashok, A. S. Dhanasekaran, A. Arunkumar, D. Stap, et al., “Super-naturalinstructions: Generalization via declarative instructions on $1 6 0 0 +$ nlp tasks,” in Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pp. 5085–5109, 2022. 





[116] S. Mishra, D. Khashabi, C. Baral, and H. Hajishirzi, “Cross-task generalization via natural language crowdsourcing instructions,” in Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 3470–3487, 2022. 





[117] Y. Wang, Y. Kordi, S. Mishra, A. Liu, N. A. Smith, D. Khashabi, and H. Hajishirzi, “Self-instruct: Aligning language model with self generated instructions,” arXiv preprint arXiv:2212.10560, 2022. 





[118] L. Weidinger, J. Mellor, M. Rauh, C. Griffin, J. Uesato, P.-S. Huang, M. Cheng, M. Glaese, B. Balle, A. Kasirzadeh, et al., “Ethical and social risks of harm from language models,” arXiv preprint arXiv:2112.04359, 2021. 





[119] S. Kiegeland and J. Kreutzer, “Revisiting the weaknesses of reinforcement learning for neural machine translation,” in Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pp. 1673–1681, 2021. 





[120] N. Jaques, J. H. Shen, A. Ghandeharioun, C. Ferguson, A. Lapedriza, N. Jones, S. Gu, and R. Picard, “Human-centric dialog training via offline reinforcement learning,” in Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 3985–4003, 2020. 





[121] S. J. Rennie, E. Marcheret, Y. Mroueh, J. Ross, and V. Goel, “Self-critical sequence training for image captioning,” in Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 7008–7024, 2017. 





[122] R. Y. Pang and H. He, “Text generation by learning from demonstrations,” in Proceedings of the international conference on learning representations, 2021. 





[123] M. Hausknecht, P. Ammanabrolu, M.-A. Côté, and X. Yuan, “Interactive fiction games: A colossal adventure,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 34, pp. 7903–7910, 2020. 





[124] C. Snell, I. Kostrikov, Y. Su, M. Yang, and S. Levine, “Offline rl for natural language generation with implicit language q learning,” arXiv preprint arXiv:2206.11871, 2022. 





[125] X. Lu, S. Welleck, L. Jiang, J. Hessel, L. Qin, P. West, P. Ammanabrolu, and Y. Choi, “Quark: Controllable text generation with reinforced unlearning,” arXiv preprint arXiv:2205.13636, 2022. 





[126] V. Uc-Cetina, N. Navarro-Guerrero, A. Martin-Gonzalez, C. Weber, and S. Wermter, “Survey on reinforcement learning for language processing,” Artificial Intelligence Review, pp. 1–33, 2022. 





[127] R. Ramamurthy, P. Ammanabrolu, K. Brantley, J. Hessel, R. Sifa, C. Bauckhage, H. Hajishirzi, and Y. Choi, “Is reinforcement learning (not) for natural language processing?: Benchmarks, baselines, and building blocks for natural language policy optimization,” arXiv preprint arXiv:2210.01241, 2022. 





[128] J. Wu, L. Ouyang, D. M. Ziegler, N. Stiennon, R. Lowe, J. Leike, and P. Christiano, “Recursively summarizing books with human feedback,” arXiv preprint arXiv:2109.10862, 2021. 





[129] R. Nakano, J. Hilton, S. Balaji, J. Wu, L. Ouyang, C. Kim, C. Hesse, S. Jain, V. Kosaraju, W. Saunders, et al., “Webgpt: Browser-assisted question-answering with human feedback,” arXiv preprint arXiv:2112.09332, 2021. 





[130] A. Glaese, N. McAleese, M. Tr˛ebacz, J. Aslanides, V. Firoiu, T. Ewalds, M. Rauh, L. Weidinger, M. Chadwick, P. Thacker, et al., “Improving alignment of dialogue agents via targeted human judgements,” arXiv preprint arXiv:2209.14375, 2022. 





[131] Y. Bai, S. Kadavath, S. Kundu, A. Askell, J. Kernion, A. Jones, A. Chen, A. Goldie, A. Mirhoseini, C. McKinnon, et al., “Constitutional ai: Harmlessness from ai feedback,” arXiv preprint arXiv:2212.08073, 2022. 





[132] H. W. Chung, L. Hou, S. Longpre, B. Zoph, Y. Tay, W. Fedus, E. Li, X. Wang, M. Dehghani, S. Brahma, et al., “Scaling instruction-finetuned language models,” arXiv preprint arXiv:2210.11416, 2022. 





[133] T. Kojima, S. S. Gu, M. Reid, Y. Matsuo, and Y. Iwasawa, “Large language models are zero-shot reasoners,” in Advances in Neural Information Processing Systems (A. H. Oh, A. Agarwal, D. Belgrave, and K. Cho, eds.), 2022. 





[134] A. Dosovitskiy, J. T. Springenberg, M. Riedmiller, and T. Brox, “Discriminative unsupervised feature learning with convolutional neural networks,” Advances in neural information processing systems, 2014. 





[135] A. Dosovitskiy, P. Fischer, J. T. Springenberg, M. Riedmiller, and T. Brox, “Discriminative unsupervised feature learning with exemplar convolutional neural networks,” TPAMI, 2016. 





[136] C. Doersch, A. Gupta, and A. A. Efros, “Unsupervised visual representation learning by context prediction,” in ICCV. 





[137] D. Pathak, P. Krahenbuhl, J. Donahue, T. Darrell, and A. A. Efros, “Context encoders: Feature learning by inpainting,” in CVPR. 





[138] R. Zhang, P. Isola, and A. A. Efros, “Colorful image colorization,” in ECCV, 2016. 





[139] R. Zhang, P. Isola, and A. A. Efros, “Split-brain autoencoders: Unsupervised learning by crosschannel prediction,” in CVPR. 





[140] M. Noroozi and P. Favaro, “Unsupervised learning of visual representations by solving jigsaw puzzles,” in ECCV. 





[141] D. Kim, D. Cho, D. Yoo, and I. S. Kweon, “Learning image representations by completing damaged jigsaw puzzles,” in WACV. 





[142] M. Noroozi, H. Pirsiavash, and P. Favaro, “Representation learning by learning to count,” in ICCV. 





[143] P. Bojanowski and A. Joulin, “Unsupervised learning by predicting noise,” in ICML, 2017. 





[144] S. Gidaris, P. Singh, and N. Komodakis, “Unsupervised representation learning by predicting image rotations,” arXiv, 2018. 





[145] A. v. d. Oord, Y. Li, and O. Vinyals, “Representation learning with contrastive predictive coding,” arXiv, 2018. 





[146] O. Henaff, “Data-efficient image recognition with contrastive predictive coding,” in ICML, 2020. 





[147] J. Donahue and K. Simonyan, “Large scale adversarial representation learning,” in NeurIPS. 





[148] V. Dumoulin, I. Belghazi, B. Poole, O. Mastropietro, A. Lamb, M. Arjovsky, and A. Courville, “Adversarially learned inference,” arXiv, 2016. 





[149] M. Chen, A. Radford, R. Child, J. Wu, H. Jun, D. Luan, and I. Sutskever, “Generative pretraining from pixels,” in ICML, 2020. 





[150] H. Bao, L. Dong, S. Piao, and F. Wei, “Beit: Bert pre-training of image transformers,” in International Conference on Learning Representations, 2021. 





[151] M. Caron, H. Touvron, I. Misra, H. Jégou, J. Mairal, P. Bojanowski, and A. Joulin, “Emerging properties in self-supervised vision transformers,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 9650–9660, 2021. 





[152] A. Ramesh, M. Pavlov, G. Goh, S. Gray, C. Voss, A. Radford, M. Chen, and I. Sutskever, “Zeroshot text-to-image generation,” in International Conference on Machine Learning, pp. 8821–8831, PMLR, 2021. 





[153] Z. Wu, Y. Xiong, S. X. Yu, and D. Lin, “Unsupervised feature learning via non-parametric instance discrimination,” in CVPR. 





[154] K. He, X. Chen, S. Xie, Y. Li, P. Dollár, and R. Girshick, “Masked autoencoders are scalable vision learners,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 16000–16009, 2022. 





[155] Z. Xie, Z. Zhang, Y. Cao, Y. Lin, J. Bao, Z. Yao, Q. Dai, and H. Hu, “Simmim: A simple framework for masked image modeling,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9653–9663, 2022. 





[156] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo, et al., “Segment anything,” arXiv preprint arXiv:2304.02643, 2023. 





[157] Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and B. Guo, “Swin transformer: Hierarchical vision transformer using shifted windows,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 10012–10022, 2021. 





[158] X. Li, W. Wang, L. Yang, and J. Yang, “Uniform masking: Enabling mae pre-training for pyramidbased vision transformers with locality,” arXiv preprint arXiv:2205.10063, 2022. 





[159] J. Chen, M. Hu, B. Li, and M. Elhoseiny, “Efficient self-supervised vision pretraining with local masked reconstruction,” arXiv preprint arXiv:2206.00790, 2022. 





[160] G. Hinton, O. Vinyals, and J. Dean, “Distilling the knowledge in a neural network,” arXiv, 2015. 





[161] C. Zhuang, A. L. Zhai, and D. Yamins, “Local aggregation for unsupervised learning of visual embeddings,” in ICCV. 





[162] I. Misra and L. v. d. Maaten, “Self-supervised learning of pretext-invariant representations,” in CVPR. 





[163] K. He, H. Fan, Y. Wu, S. Xie, and R. Girshick, “Momentum contrast for unsupervised visual representation learning,” in CVPR. 





[164] X. Chen, H. Fan, R. Girshick, and K. He, “Improved baselines with momentum contrastive learning,” arXiv, 2020. 





[165] J.-B. Grill, F. Strub, F. Altché, C. Tallec, P. H. Richemond, E. Buchatskaya, C. Doersch, B. A. Pires, Z. D. Guo, M. G. Azar, et al., “Bootstrap your own latent: A new approach to self-supervised learning,” arXiv, 2020. 





[166] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, “A simple framework for contrastive learning of visual representations,” in ICML. 





[167] M. Caron, I. Misra, J. Mairal, P. Goyal, P. Bojanowski, and A. Joulin, “Unsupervised learning of visual features by contrasting cluster assignments,” arXiv, 2020. 





[168] P. Goyal, M. Caron, B. Lefaudeux, M. Xu, P. Wang, V. Pai, M. Singh, V. Liptchinsky, I. Misra, A. Joulin, et al., “Self-supervised pretraining of visual features in the wild,” arXiv, 2021. 





[169] I. Radosavovic, R. P. Kosaraju, R. Girshick, K. He, and P. Dollár, “Designing network design spaces,” in CVPR. 





[170] X. Chen and K. He, “Exploring simple siamese representation learning,” in CVPR. 





[171] J. Li, P. Zhou, C. Xiong, and S. C. H. Hoi, “Prototypical contrastive learning of unsupervised representations,” in ICLR, OpenReview.net, 2021. 





[172] L. Zhang, G.-J. Qi, L. Wang, and J. Luo, “Aet vs. aed: Unsupervised representation learning by auto-encoding transformations rather than data,” in CVPR. 





[173] P. Bachman, R. D. Hjelm, and W. Buchwalter, “Learning representations by maximizing mutual information across views,” arXiv, 2019. 





[174] X. Yan, I. Misra, A. Gupta, D. Ghadiyaram, and D. Mahajan, “Clusterfit: Improving generalization of visual representations,” in CVPR, 2020. 





[175] Y. M. Asano, C. Rupprecht, and A. Vedaldi, “Self-labelling via simultaneous clustering and representation learning,” arXiv preprint arXiv:1911.05371, 2019. 





[176] T. Chen, S. Kornblith, K. Swersky, M. Norouzi, and G. Hinton, “Big self-supervised models are strong semi-supervised learners,” arXiv, 2020. 





[177] Y. Tian, D. Krishnan, and P. Isola, “Contrastive multiview coding,” arXiv, 2019. 





[178] E. D. Cubuk, B. Zoph, J. Shlens, and Q. V. Le, “Randaugment: Practical data augmentation with no separate search,” arXiv, 2019. 





[179] Y. Tian, C. Sun, B. Poole, D. Krishnan, C. Schmid, and P. Isola, “What makes for good views for contrastive learning,” arXiv, 2020. 





[180] X. Chen, S. Xie, and K. He, “An empirical study of training self-supervised vision transformers,” arXiv, 2021. 





[181] J. Mitrovic, B. McWilliams, J. C. Walker, L. H. Buesing, and C. Blundell, “Representation learning via invariant causal mechanisms,” in ICLR. 





[182] Y. Tian, X. Chen, and S. Ganguli, “Understanding self-supervised learning dynamics without contrastive pairs,” in ICML. 





[183] Z. Xie, Y. Lin, Z. Yao, Z. Zhang, Q. Dai, Y. Cao, and H. Hu, “Self-supervised learning with swin transformers,” arXiv, 2021. 





[184] Z. Li, Z. Chen, F. Yang, W. Li, Y. Zhu, C. Zhao, R. Deng, L. Wu, R. Zhao, M. Tang, et al., “Mst: Masked self-supervised transformer for visual representation,” Advances in Neural Information Processing Systems, vol. 34, 2021. 





[185] H. Bao, L. Dong, S. Piao, and F. Wei, “BEit: BERT pre-training of image transformers,” in International Conference on Learning Representations, 2022. 





[186] X. Chen, M. Ding, X. Wang, Y. Xin, S. Mo, Y. Wang, S. Han, P. Luo, G. Zeng, and J. Wang, “Context autoencoder for self-supervised representation learning,” arXiv preprint arXiv:2202.03026, 2022. 





[187] X. Dong, J. Bao, T. Zhang, D. Chen, W. Zhang, L. Yuan, D. Chen, F. Wen, and N. Yu, “Peco: Perceptual codebook for bert pre-training of vision transformers,” arXiv preprint arXiv:2111.12710, 2021. 





[188] Y. You, T. Chen, Z. Wang, and Y. Shen, “When does self-supervision help graph convolutional networks?,” in ICML. 





[189] W. Jin, T. Derr, H. Liu, Y. Wang, S. Wang, Z. Liu, and J. Tang, “Self-supervised learning on graphs: Deep insights and new direction,” CoRR, 2020. 





[190] W. Hu, B. Liu, J. Gomes, M. Zitnik, P. Liang, V. S. Pande, and J. Leskovec, “Strategies for pre-training graph neural networks,” in ICLR, 2020. 





[191] B. Perozzi, R. Al-Rfou, and S. Skiena, “Deepwalk: online learning of social representations,” in ACM SIGKDD. 





[192] A. Grover and J. Leskovec, “node2vec: Scalable feature learning for networks,” in ACM SIGKDD. 





[193] J. Tang, M. Qu, M. Wang, M. Zhang, J. Yan, and Q. Mei, “LINE: large-scale information network embedding,” in WWW. 





[194] T. N. Kipf and M. Welling, “Variational graph auto-encoders,” CoRR, 2016. 





[195] J. Qiu, Q. Chen, Y. Dong, J. Zhang, H. Yang, M. Ding, K. Wang, and J. Tang, “GCC: graph contrastive coding for graph neural network pre-training,” in KDD. 





[196] Y. Zhu, Y. Xu, F. Yu, Q. Liu, S. Wu, and L. Wang, “Graph contrastive learning with adaptive augmentation,” in WWW, 2021. 





[197] Y. You, T. Chen, Y. Sui, T. Chen, Z. Wang, and Y. Shen, “Graph contrastive learning with augmentations,” in NeurIPS, 2020. 





[198] C. Mavromatis and G. Karypis, “Graph infoclust: Maximizing coarse-grain mutual information in graphs,” in PAKDD, 2021. 





[199] Q. Sun, J. Li, H. Peng, J. Wu, Y. Ning, P. S. Yu, and L. He, “SUGAR: subgraph neural network with reinforcement pooling and self-supervised mutual information mechanism,” in WWW, 2021. 





[200] P. Velickovic, W. Fedus, W. L. Hamilton, P. Liò, Y. Bengio, and R. D. Hjelm, “Deep graph infomax,” in ICLR, 2019. 





[201] K. Hassani and A. H. K. Ahmadi, “Contrastive multi-view representation learning on graphs,” in ICML. 





[202] Y. Jiao, Y. Xiong, J. Zhang, Y. Zhang, T. Zhang, and Y. Zhu, “Sub-graph contrast for scalable selfsupervised graph representation learning,” in ICDM. 





[203] K. Sun, Z. Lin, and Z. Zhu, “Multi-stage self-supervised learning for graph convolutional networks on graphs with few labeled nodes,” in AAAI. 





[204] Y. Rong, Y. Bian, T. Xu, W. Xie, Y. Wei, W. Huang, and J. Huang, “Self-supervised graph transformer on large-scale molecular data,” in NeurIPS, 2020. 





[205] Q. Tan, N. Liu, X. Huang, R. Chen, S.-H. Choi, and X. Hu, “Mgae: Masked autoencoders for selfsupervised learning on graphs,” arXiv preprint arXiv:2201.02534, 2022. 





[206] Z. Hou, X. Liu, Y. Dong, C. Wang, J. Tang, et al., “Graphmae: Self-supervised masked graph autoencoders,” arXiv preprint arXiv:2205.10803, 2022. 





[207] J. Li, R. Wu, W. Sun, L. Chen, S. Tian, L. Zhu, C. Meng, Z. Zheng, and W. Wang, “Maskgae: Masked graph modeling meets graph autoencoders,” arXiv preprint arXiv:2205.10053, 2022. 





[208] Y. Tian, K. Dong, C. Zhang, C. Zhang, and N. V. Chawla, “Heterogeneous graph masked autoencoders,” arXiv preprint arXiv:2208.09957, 2022. 





[209] S. Wan, S. Pan, J. Yang, and C. Gong, “Contrastive and generative graph convolutional networks for graph-based semi-supervised learning,” in AAAI. 





[210] J. Zhang, H. Zhang, C. Xia, and L. Sun, “Graph-bert: Only attention is needed for learning graph representations,” arXiv, 2020. 





[211] Z. Peng, W. Huang, M. Luo, Q. Zheng, Y. Rong, T. Xu, and J. Huang, “Graph representation learning via graphical mutual information maximization,” in WWW, 2020. 





[212] Z. Hu, Y. Dong, K. Wang, K. Chang, and Y. Sun, “GPT-GNN: generative pre-training of graph neural networks,” in KDD. 





[213] G. Wang, H. Guo, A. Li, X. Liu, and Q. Yan, “Federated iot interaction vulnerability analysis,” in 2023 IEEE 39th International Conference on Data Engineering (ICDE), IEEE, 2023. 





[214] W. L. Hamilton, Z. Ying, and J. Leskovec, “Inductive representation learning on large graphs,” in NIPS, 2017. 





[215] D. Hwang, J. Park, S. Kwon, K. Kim, J. Ha, and H. J. Kim, “Self-supervised auxiliary learning with meta-paths for heterogeneous graphs,” in NeurIPS, 2020. 





[216] F. Sun, J. Hoffmann, V. Verma, and J. Tang, “Infograph: Unsupervised and semi-supervised graphlevel representation learning via mutual information maximization,” in ICLR, 2020. 





[217] C. Park, D. Kim, J. Han, and H. Yu, “Unsupervised attributed multiplex network embedding,” in AAAI. 





[218] Y. You, T. Chen, Y. Shen, and Z. Wang, “Graph contrastive learning automated,” CoRR, 2021. 





[219] J. Zeng and P. Xie, “Contrastive self-supervised learning for graph classification,” in AAAI. 





[220] M. Xu, H. Wang, B. Ni, H. Guo, and J. Tang, “Self-supervised graph-level representation learning with local and global structure,” CoRR, 2021. 





[221] P. Wang, K. Agarwal, C. Ham, S. Choudhury, and C. K. Reddy, “Self-supervised learning of contextual embeddings for link prediction in heterogeneous networks,” in WWW. 





[222] J. Cao, X. Lin, S. Guo, L. Liu, T. Liu, and B. Wang, “Bipartite graph embedding via mutual information maximization,” in WSDM, 2021. 





[223] X. Wang, N. Liu, H. Han, and C. Shi, “Self-supervised heterogeneous graph neural network with co-contrastive learning,” KDD, 2021. 





[224] D. Kim and A. Oh, “How to find your friendly neighborhood: Graph attention design with selfsupervision,” in ICLR, 2021. 





[225] M. Sun, J. Xing, H. Wang, B. Chen, and J. Zhou, “Mocl: Contrastive learning on molecular graphs with multi-level domain knowledge,” CoRR, 2021. 





[226] S. Schneider, A. Baevski, R. Collobert, and M. Auli, “wav2vec: Unsupervised pre-training for speech recognition,” in INTERSPEECH. 





[227] A. Baevski, S. Schneider, and M. Auli, “vq-wav2vec: Self-supervised learning of discrete speech representations,” in ICLR. 





[228] A. Baevski, Y. Zhou, A. Mohamed, and M. Auli, “wav2vec 2.0: A framework for self-supervised learning of speech representations,” in NeurIPS, 2020. 





[229] Y. Chung and J. R. Glass, “Generative pre-training for speech with autoregressive predictive coding,” in ICASSP. 





[230] X. Song, G. Wang, Y. Huang, Z. Wu, D. Su, and H. Meng, “Speech-xlnet: Unsupervised acoustic model pretraining for self-attention networks,” in INTERSPEECH. 





[231] Y. Chung, Y. Wang, W. Hsu, Y. Zhang, and R. J. Skerry-Ryan, “Semi-supervised training for improving data efficiency in end-to-end speech synthesis,” in ICASSP, 2019. 





[232] P. Denisov and N. T. Vu, “Pretrained semantic speech embeddings for end-to-end spoken language understanding via cross-modal teacher-student learning,” in Interspeech, 2020. 





[233] Y.-A. Chung, C. Zhu, and M. Zeng, “SPLAT: Speech-language joint pre-training for spoken language understanding,” in ACL, 2021. 





[234] M. Zeng, X. Tan, R. Wang, Z. Ju, T. Qin, and T.-Y. Liu, “Musicbert: Symbolic music understanding with large-scale pre-training,” arXiv preprint arXiv:2106.05630, 2021. 





[235] Y.-S. Huang and Y.-H. Yang, “Pop music transformer: Beat-based modeling and generation of expressive pop piano compositions,” in Proceedings of the 28th ACM International Conference on Multimedia, pp. 1180–1188, 2020. 





[236] P. Verma and J. Berger, “Audio transformers: Transformer architectures for large scale audio understanding. adieu convolutions,” arXiv preprint arXiv:2105.00335, 2021. 





[237] B. Fernando, H. Bilen, E. Gavves, and S. Gould, “Self-supervised video representation learning with odd-one-out networks,” in CVPR. 





[238] I. Misra, C. L. Zitnick, and M. Hebert, “Shuffle and learn: unsupervised learning using temporal order verification,” in ECCV, 2016. 





[239] D. Kim, D. Cho, and I. S. Kweon, “Self-supervised video representation learning with space-time cubic puzzles,” in AAAI. 





[240] L. Tao, X. Wang, and T. Yamasaki, “Self-supervised video representation learning using inter-intra contrastive framework,” in ACM Multimedia. 





[241] G. Lorre, J. Rabarisoa, A. Orcesi, S. Ainouz, and S. Canu, “Temporal contrastive pretraining for video action recognition,” in WACV. 





[242] T. Yao, Y. Zhang, Z. Qiu, Y. Pan, and T. Mei, “Seco: Exploring sequence supervision for unsupervised representation learning,” arXiv, 2020. 





[243] L. H. Li, M. Yatskar, D. Yin, C. Hsieh, and K. Chang, “Visualbert: A simple and performant baseline for vision and language,” CoRR, 2019. 





[244] G. Li, N. Duan, Y. Fang, M. Gong, and D. Jiang, “Unicoder-vl: A universal encoder for vision and language by cross-modal pre-training,” in AAAI. 





[245] W. Su, X. Zhu, Y. Cao, B. Li, L. Lu, F. Wei, and J. Dai, “VL-BERT: pre-training of generic visuallinguistic representations,” in ICLR, 2020. 





[246] J. Lu, D. Batra, D. Parikh, and S. Lee, “Vilbert: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks,” in NeurIPS. 





[247] A. Ramesh, P. Dhariwal, A. Nichol, C. Chu, and M. Chen, “Hierarchical text-conditional image generation with clip latents,” arXiv preprint arXiv:2204.06125, 2022. 





[248] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, et al., “Learning transferable visual models from natural language supervision,” in International Conference on Machine Learning, pp. 8748–8763, PMLR, 2021. 





[249] A. Nichol, P. Dhariwal, A. Ramesh, P. Shyam, P. Mishkin, B. McGrew, I. Sutskever, and M. Chen, “Glide: Towards photorealistic image generation and editing with text-guided diffusion models,” arXiv preprint arXiv:2112.10741, 2021. 





[250] N. Sayed, B. Brattoli, and B. Ommer, “Cross and learn: Cross-modal self-supervision,” in GCPR, 2018. 





[251] Z. Ren and Y. J. Lee, “Cross-domain self-supervised multi-task feature learning using synthetic imagery,” in CVPR. 





[252] Y. Tian, D. Krishnan, and P. Isola, “Contrastive multiview coding,” in ECCV. 





[253] A. Zlotchevski, D. Drain, A. Svyatkovskiy, C. B. Clement, N. Sundaresan, and M. Tufano, “Exploring and evaluating personalized models for code generation,” in Proceedings of the 30th ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering, pp. 1500–1508, 2022. 





[254] S. Thakur, B. Ahmad, Z. Fan, H. Pearce, B. Tan, R. Karri, B. Dolan-Gavitt, and S. Garg, “Benchmarking large language models for automated verilog rtl code generation,” arXiv preprint arXiv:2212.11140, 2022. 





[255] E. Nijkamp, B. Pang, H. Hayashi, L. Tu, H. Wang, Y. Zhou, S. Savarese, and C. Xiong, “Codegen: An open large language model for code with multi-turn program synthesis,” arXiv preprint arXiv:2203.13474, 2022. 





[256] G. Poesia, O. Polozov, V. Le, A. Tiwari, G. Soares, C. Meek, and S. Gulwani, “Synchromesh: Reliable code generation from pre-trained language models,” arXiv preprint arXiv:2201.11227, 2022. 





[257] Y.-C. Chen, L. Li, L. Yu, A. El Kholy, F. Ahmed, Z. Gan, Y. Cheng, and J. Liu, “Uniter: Universal image-text representation learning,” in European conference on computer vision, pp. 104–120, Springer, 2020. 





[258] X. Zhu, J. Zhu, H. Li, X. Wu, H. Li, X. Wang, and J. Dai, “Uni-perceiver: Pre-training unified architecture for generic perception for zero-shot and few-shot tasks,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 16804–16815, 2022. 





[259] S. Reed, K. Zolna, E. Parisotto, S. G. Colmenarejo, A. Novikov, G. Barth-Maron, M. Gimenez, Y. Sulsky, J. Kay, J. T. Springenberg, et al., “A generalist agent,” arXiv preprint arXiv:2205.06175, 2022. 





[260] W. Li, C. Gao, G. Niu, X. Xiao, H. Liu, J. Liu, H. Wu, and H. Wang, “Unimo: Towards unified-modal understanding and generation via cross-modal contrastive learning,” arXiv preprint arXiv:2012.15409, 2020. 





[261] V. Sanh, L. Debut, J. Chaumond, and T. Wolf, “Distilbert, a distilled version of BERT: smaller, faster, cheaper and lighter,” CoRR, 2019. 





[262] D. Jin, Z. Jin, J. T. Zhou, and P. Szolovits, “Is bert really robust? a strong baseline for natural language attack on text classification and entailment,” in AAAI. 





[263] D. Jin, Z. Jin, J. T. Zhou, and P. Szolovits, “Is BERT really robust? A strong baseline for natural language attack on text classification and entailment,” in AAAI. 





[264] Y. Zang, F. Qi, C. Yang, Z. Liu, M. Zhang, Q. Liu, and M. Sun, “Word-level textual adversarial attacking as combinatorial optimization,” in ACL. 





[265] E. Wallace, S. Feng, N. Kandpal, M. Gardner, and S. Singh, “Universal adversarial triggers for attacking and analyzing NLP,” in EMNLP-IJCNLP. 





[266] K. Kurita, P. Michel, and G. Neubig, “Weight poisoning attacks on pretrained models,” in ACL. 





[267] R. Schuster, T. Schuster, Y. Meri, and V. Shmatikov, “Humpty dumpty: Controlling word meanings via corpus poisoning,” in IEEE Symposium on Security and Privacy, 2020. 





[268] R. Bao, J. Wang, and H. Zhao, “Defending pre-trained language models from adversarial word substitution without performance sacrifice,” in ACL/IJCNLP. 





[269] Z. Zhang, Y. Li, J. Wang, B. Liu, D. Li, Y. Guo, X. Chen, and Y. Liu, “Remos: reducing defect inheritance in transfer learning via relevant model slicing,” in Proceedings of the 44th International Conference on Software Engineering, pp. 1856–1868, 2022. 





[270] N. Carlini, F. Tramer, E. Wallace, M. Jagielski, A. Herbert-Voss, K. Lee, A. Roberts, T. B. Brown, D. Song, U. Erlingsson, et al., “Extracting training data from large language models.,” in USENIX Security Symposium, vol. 6, 2021. 





[271] G. Wang, L. Zhang, Z. Yang, and X.-Y. Li, “Socialite: Social activity mining and friend autolabeling,” in 2018 IEEE 37th International Performance Computing and Communications Conference (IPCCC), pp. 1–8, IEEE, 2018. 





[272] F. Han, L. Zhang, X. You, G. Wang, and X.-Y. Li, “Shad: Privacy-friendly shared activity detection and data sharing,” in 2019 IEEE 16th International Conference on Mobile Ad Hoc and Sensor Systems (MASS), pp. 109–117, IEEE, 2019. 





[273] T. Chen, X. Zhai, M. Ritter, M. Lucic, and N. Houlsby, “Self-supervised gans via auxiliary rotation loss,” in CVPR, 2019. 





[274] S. Abnar, M. Dehghani, B. Neyshabur, and H. Sedghi, “Exploring the limits of large scale pretraining,” arXiv, 2021. 





[275] M. E. Peters, M. Neumann, M. Iyyer, M. Gardner, C. Clark, K. Lee, and L. Zettlemoyer, “Deep contextualized word representations,” in NAACL-HLT, 2018. 





[276] T. Mikolov, I. Sutskever, K. Chen, G. S. Corrado, and J. Dean, “Distributed representations of words and phrases and their compositionality,” in NIPS. 





[277] R. Collobert, J. Weston, L. Bottou, M. Karlen, K. Kavukcuoglu, and P. P. Kuksa, “Natural language processing (almost) from scratch,” J. Mach. Learn. Res., 2011. 





[278] A. Neelakantan, J. Shankar, A. Passos, and A. McCallum, “Efficient non-parametric estimation of multiple embeddings per word in vector space,” in EMNLP. 





[279] P. Zhou, Z. Qi, S. Zheng, J. Xu, H. Bao, and B. Xu, “Text classification improved by integrating bidirectional LSTM with two-dimensional max pooling,” in COLING. 





[280] D. U. Hui, X. U. Xueke, W. U. Dayong, Y. Liu, Y. U. Zhihua, and X. Cheng, “A sentiment classification method based on sentiment-specific word embedding,” Journal of Chinese Information Processing, 2017. 





[281] Y. Liu, C. Ma, and Y. Zhang, “Hierarchical machine translation model based on deep recursive neural network,” Chin. J. Comput, 2017. 





[282] X. Liang, F. Ren, Y. Liu, L. Pan, Y. Hou, Y. Zhang, and L. I. Yan, “N-reader: Machine reading comprehension model based on double layers of self-attention,” Journal of Chinese Information Processing, 2018. 





[283] Z. Zhichang, Z. Zhenwen, and Z. Zhiman, “User intent classification based on indrnn-attention,” Journal of Computer Research and Development, 2019. 





[284] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, “Imagenet: A large-scale hierarchical image database,” in 2009 IEEE conference on computer vision and pattern recognition, 2009. 





[285] A. Krizhevsky, I. Sutskever, and G. E. Hinton, “Imagenet classification with deep convolutional neural networks,” Advances in neural information processing systems, 2012. 





[286] M. Lin, Q. Chen, and S. Yan, “Network in network,” arXiv, 2013. 





[287] K. Simonyan and A. Zisserman, “Very deep convolutional networks for large-scale image recognition,” arXiv, 2014. 





[288] C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich, “Going deeper with convolutions,” in CVPR. 





[289] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in CVPR. 





[290] G. Huang, Z. Liu, L. Van Der Maaten, and K. Q. Weinberger, “Densely connected convolutional networks,” in CVPR. 





[291] R. Girshick, J. Donahue, T. Darrell, and J. Malik, “Rich feature hierarchies for accurate object detection and semantic segmentation,” in CVPR. 





[292] R. Girshick, “Fast r-cnn,” in ICCV. 





[293] S. Ren, K. He, R. Girshick, and J. Sun, “Faster r-cnn: Towards real-time object detection with region proposal networks,” arXiv, 2015. 





[294] K. He, G. Gkioxari, P. Dollár, and R. Girshick, “Mask r-cnn,” in ICCV. 





[295] J. Long, E. Shelhamer, and T. Darrell, “Fully convolutional networks for semantic segmentation,” in CVPR. 





[296] W. Liu, D. Anguelov, D. Erhan, C. Szegedy, S. Reed, C.-Y. Fu, and A. C. Berg, “Ssd: Single shot multibox detector,” in European conference on computer vision, 2016. 





[297] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, “You only look once: Unified, real-time object detection,” in CVPR. 





[298] J. Redmon and A. Farhadi, “Yolo9000: better, faster, stronger,” in CVPR. 





[299] J. Redmon and A. Farhadi, “Yolov3: An incremental improvement,” arXiv, 2018. 





[300] A. Bochkovskiy, C.-Y. Wang, and H.-Y. M. Liao, “Yolov4: Optimal speed and accuracy of object detection,” arXiv, 2020. 





[301] Q. Chen, Y. Wang, T. Yang, X. Zhang, J. Cheng, and J. Sun, “You only look one-level feature,” arXiv, 2021. 





[302] V. Badrinarayanan, A. Kendall, and R. Cipolla, “Segnet: A deep convolutional encoder-decoder architecture for image segmentation,” IEEE transactions on pattern analysis and machine intelligence, 2017. 





[303] H. Zhao, J. Shi, X. Qi, X. Wang, and J. Jia, “Pyramid scene parsing network,” in CVPR, 2017. 





[304] L.-C. Chen, G. Papandreou, I. Kokkinos, K. Murphy, and A. L. Yuille, “Semantic image segmentation with deep convolutional nets and fully connected crfs,” arXiv, 2014. 





[305] L.-C. Chen, G. Papandreou, I. Kokkinos, K. Murphy, and A. L. Yuille, “Deeplab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected crfs,” IEEE transactions on pattern analysis and machine intelligence, 2017. 





[306] L.-C. Chen, G. Papandreou, F. Schroff, and H. Adam, “Rethinking atrous convolution for semantic image segmentation,” arXiv, 2017. 





[307] L.-C. Chen, Y. Zhu, G. Papandreou, F. Schroff, and H. Adam, “Encoder-decoder with atrous separable convolution for semantic image segmentation,” in ECCV. 





[308] G. Lin, A. Milan, C. Shen, and I. Reid, “Refinenet: Multi-path refinement networks for highresolution semantic segmentation,” in CVPR. 





[309] M. Everingham, S. A. Eslami, L. Van Gool, C. K. Williams, J. Winn, and A. Zisserman, “The pascal visual object classes challenge: A retrospective,” International journal of computer vision, 2015. 





[310] M. Everingham, L. Van Gool, C. K. Williams, J. Winn, and A. Zisserman, “The pascal visual object classes (voc) challenge,” International journal of computer vision, 2010. 





[311] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollár, and C. L. Zitnick, “Microsoft coco: Common objects in context,” in European conference on computer vision, 2014. 





[312] D. E. Rumelhart, G. E. Hinton, and R. J. Williams, “Learning representations by back-propagating errors,” nature, 1986. 





[313] M. I. Jordan, “Serial order: A parallel distributed processing approach,” in Advances in psychology, 1997. 





[314] J. L. Elman, “Finding structure in time,” Cognitive science, 1990. 





[315] S. Hochreiter and J. Schmidhuber, “Long short-term memory,” Neural Computation, 1997. 





[316] A. Graves, M. Liwicki, S. Fernández, R. Bertolami, H. Bunke, and J. Schmidhuber, “A novel connectionist system for unconstrained handwriting recognition,” IEEE transactions on pattern analysis and machine intelligence, 2008. 





[317] O. Vinyals, A. Toshev, S. Bengio, and D. Erhan, “Show and tell: A neural image caption generator,” in CVPR. 





[318] I. Sutskever, O. Vinyals, and Q. V. Le, “Sequence to sequence learning with neural networks,” arXiv, 2014. 





[319] A. Graves, “Generating sequences with recurrent neural networks,” arXiv, 2013. 





[320] M. Sundermeyer, R. Schlüter, and H. Ney, “Lstm neural networks for language modeling,” in INTER-SPEECH, 2012. 





[321] I. J. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio, “Generative adversarial networks,” arXiv, 2014. 





[322] P. Isola, J.-Y. Zhu, T. Zhou, and A. A. Efros, “Image-to-image translation with conditional adversarial networks,” in CVPR, 2017. 





[323] C. Li and M. Wand, “Precomputed real-time texture synthesis with markovian generative adversarial networks,” in European conference on computer vision, 2016. 





[324] J.-Y. Zhu, T. Park, P. Isola, and A. A. Efros, “Unpaired image-to-image translation using cycleconsistent adversarial networks,” in ICCV. 





[325] T. Karras, S. Laine, and T. Aila, “A style-based generator architecture for generative adversarial networks,” in CVPR, 2019. 





[326] A. Van Oord, N. Kalchbrenner, and K. Kavukcuoglu, “Pixel recurrent neural networks,” in International Conference on Machine Learning, 2016. 





[327] T. Kim, M. Cha, H. Kim, J. K. Lee, and J. Kim, “Learning to discover cross-domain relations with generative adversarial networks,” in International Conference on Machine Learning, 2017. 





[328] E. Denton, S. Chintala, A. Szlam, and R. Fergus, “Deep generative image models using a laplacian pyramid of adversarial networks,” arXiv, 2015. 





[329] X. Huang, Y. Li, O. Poursaeed, J. Hopcroft, and S. Belongie, “Stacked generative adversarial networks,” in CVPR. 





[330] J. Hu, L. Shen, and G. Sun, “Squeeze-and-excitation networks,” in CVPR. 





[331] S. Woo, J. Park, J.-Y. Lee, and I. S. Kweon, “Cbam: Convolutional block attention module,” in ECCV. 





[332] Y. Cao, J. Xu, S. Lin, F. Wei, and H. Hu, “Gcnet: Non-local networks meet squeeze-excitation networks and beyond,” in ICCV Workshops. 





[333] Z. Huang, X. Wang, L. Huang, C. Huang, Y. Wei, and W. Liu, “Ccnet: Criss-cross attention for semantic segmentation,” in ICCV. 





[334] H. Zhang, I. Goodfellow, D. Metaxas, and A. Odena, “Self-attention generative adversarial networks,” in International conference on machine learning, 2019. 





[335] H. Touvron, M. Cord, M. Douze, F. Massa, A. Sablayrolles, and H. Jégou, “Training data-efficient image transformers & distillation through attention,” arXiv, 2020. 





[336] N. Carion, F. Massa, G. Synnaeve, N. Usunier, A. Kirillov, and S. Zagoruyko, “End-to-end object detection with transformers,” in European Conference on Computer Vision, 2020. 





[337] B. Graham, A. El-Nouby, H. Touvron, P. Stock, A. Joulin, H. Jégou, and M. Douze, “Levit: a vision transformer in convnet’s clothing for faster inference,” arXiv, 2021. 





[338] D. Zhou, B. Kang, X. Jin, L. Yang, X. Lian, Z. Jiang, Q. Hou, and J. Feng, “Deepvit: Towards deeper vision transformer,” arXiv, 2021. 





[339] W. Wang, E. Xie, X. Li, D.-P. Fan, K. Song, D. Liang, T. Lu, P. Luo, and L. Shao, “Pyramid vision transformer: A versatile backbone for dense prediction without convolutions,” arXiv, 2021. 





[340] T. Guan, J. Wang, S. Lan, R. Chandra, Z. Wu, L. Davis, and D. Manocha, “M3detr: Multirepresentation, multi-scale, mutual-relation 3d object detection with transformers,” arXiv, 2021. 





[341] J. M. J. Valanarasu, P. Oza, I. Hacihaliloglu, and V. M. Patel, “Medical transformer: Gated axialattention for medical image segmentation,” arXiv, 2021. 





[342] R. C. T. Lee, Y. H. Chin, and S. C. Chang, “Application of principal component analysis to multikey searching,” IEEE Trans. Software Eng., no. 3, pp. 185–193, 1976. 





[343] J. Ye, R. Janardan, and Q. Li, “Two-dimensional linear discriminant analysis,” in Advances in Neural Information Processing Systems 17 [Neural Information Processing Systems, NIPS 2004, December 13-18, 2004, Vancouver, British Columbia, Canada], pp. 1569–1576, 2004. 





[344] S. Robinson and R. Bennett, “A typology of deviant workplace behaviors: A multidimensional scaling study,” Academy of Management Journal, vol. 38, pp. 555–572, 1995. 





[345] O. Samko, A. D. Marshall, and P. L. Rosin, “Selection of the optimal parameter value for the isomap algorithm,” Pattern Recognit. Lett., no. 9, pp. 968–979, 2006. 





[346] S. T. Roweis and L. K. Saul, “Nonlinear dimensionality reduction by locally linear embedding,” Science, vol. 290, no. 5500, pp. 2323–2326, 2000. 





[347] M. Belkin and P. Niyogi, “Laplacian eigenmaps for dimensionality reduction and data representation,” Neural Comput., no. 6, pp. 1373–1396, 2003. 





[348] A. P. Singh and G. J. Gordon, “Relational learning via collective matrix factorization,” in ACM SIGKDD. 





[349] S. Cao, W. Lu, and Q. Xu, “Grarep: Learning graph representations with global structural information,” in CIKM. 





[350] M. Ou, P. Cui, J. Pei, Z. Zhang, and W. Zhu, “Asymmetric transitivity preserving graph embedding,” in ACM SIGKDD. 





[351] M. Sugiyama and K. M. Borgwardt, “Halting in random walk kernels,” in NIPS, 2015. 





[352] U. Kang, H. Tong, and J. Sun, “Fast random walk graph kernel,” in SIAM. 





[353] N. Shervashidze, P. Schweitzer, E. J. van Leeuwen, K. Mehlhorn, and K. M. Borgwardt, “Weisfeilerlehman graph kernels,” J. Mach. Learn. Res., pp. 2539–2561, 2011. 





[354] D. Erhan, P.-A. Manzagol, Y. Bengio, S. Bengio, and P. Vincent, “The difficulty of training deep architectures and the effect of unsupervised pre-training,” in Artificial Intelligence and Statistics, 2009. 





[355] D. Erhan, A. Courville, Y. Bengio, and P. Vincent, “Why does unsupervised pre-training help deep learning?,” in AISTATS. 





[356] J. D. Lee, Q. Lei, N. Saunshi, and J. Zhuo, “Predicting what you already know helps: Provable self-supervised learning,” arXiv, 2020. 





[357] C. Tosh, A. Krishnamurthy, and D. Hsu, “Contrastive learning, multi-view redundancy, and linear models,” in Algorithmic Learning Theory, 2021. 





[358] S. Arora, H. Khandeparkar, M. Khodak, O. Plevrakis, and N. Saunshi, “A theoretical analysis of contrastive unsupervised representation learning,” arXiv, 2019. 





[359] S. Anwar, M. Tahir, C. Li, A. Mian, F. S. Khan, and A. W. Muzaffar, “Image colorization: A survey and dataset,” arXiv, 2020. 





[360] C. Ledig, L. Theis, F. Huszár, J. Caballero, A. Cunningham, A. Acosta, A. Aitken, A. Tejani, J. Totz, Z. Wang, et al., “Photo-realistic single image super-resolution using a generative adversarial network,” in CVPR. 





[361] G. Perarnau, J. Van De Weijer, B. Raducanu, and J. M. Álvarez, “Invertible conditional gans for image editing,” arXiv, 2016. 





[362] C. Vondrick, H. Pirsiavash, and A. Torralba, “Generating videos with scene dynamics,” arXiv, 2016. 





[363] S. Tulyakov, M.-Y. Liu, X. Yang, and J. Kautz, “Mocogan: Decomposing motion and content for video generation,” in CVPR. 





[364] X. Wang and A. Gupta, “Unsupervised learning of visual representations using videos,” in ICCV. 





[365] C. Wei, L. Xie, X. Ren, Y. Xia, C. Su, J. Liu, Q. Tian, and A. L. Yuille, “Iterative reorganization with weak spatial constraints: Solving arbitrary jigsaw puzzles for unsupervised representation learning,” in CVPR. 





[366] U. Ahsan, R. Madhok, and I. Essa, “Video jigsaw: Unsupervised learning of spatiotemporal context for video action recognition,” in WACV. 





[367] D. Pathak, R. Girshick, P. Dollár, T. Darrell, and B. Hariharan, “Learning features by watching objects move,” in CVPR. 





[368] I. Croitoru, S.-V. Bogolin, and M. Leordeanu, “Unsupervised learning from video to detect foreground objects in single images,” in ICCV. 





[369] P. Sermanet, C. Lynch, Y. Chebotar, J. Hsu, E. Jang, S. Schaal, S. Levine, and G. Brain, “Timecontrastive networks: Self-supervised learning from video,” in ICRA. 





[370] B. Korbar, D. Tran, and L. Torresani, “Cooperative learning of audio and video models from selfsupervised synchronization,” arXiv, 2018. 





[371] B. C. Stadie, S. Levine, and P. Abbeel, “Incentivizing exploration in reinforcement learning with deep predictive models,” arXiv preprint arXiv:1507.00814, 2015. 





[372] J. Achiam and S. Sastry, “Surprise-based intrinsic motivation for deep reinforcement learning,” arXiv preprint arXiv:1703.01732, 2017. 





[373] D. Pathak, P. Agrawal, A. A. Efros, and T. Darrell, “Curiosity-driven exploration by self-supervised prediction,” in International conference on machine learning, pp. 2778–2787, PMLR, 2017. 





[374] H. Tang, R. Houthooft, D. Foote, A. Stooke, O. Xi Chen, Y. Duan, J. Schulman, F. DeTurck, and P. Abbeel, “# exploration: A study of count-based exploration for deep reinforcement learning,” Advances in neural information processing systems, vol. 30, 2017. 





[375] P. Dey and S. Medya, “Manipulating node similarity measures in network,” arXiv, 2019. 





[376] B. Han, C. Zheng, H. Chan, K. Paster, M. Zhang, and J. Ba, “Learning domain invariant representations in goal-conditioned block mdps,” Advances in Neural Information Processing Systems, vol. 34, pp. 764–776, 2021. 





[377] Y. Ding, C. Florensa, P. Abbeel, and M. Phielipp, “Goal-conditioned imitation learning,” Advances in neural information processing systems, vol. 32, 2019. 





[378] R. Shah and V. Kumar, “Rrl: Resnet as representation for reinforcement learning,” arXiv preprint arXiv:2107.03380, 2021. 





[379] T. Xiao, I. Radosavovic, T. Darrell, and J. Malik, “Masked visual pre-training for motor control,” arXiv preprint arXiv:2203.06173, 2022. 





[380] M. Schwarzer, N. Rajkumar, M. Noukhovitch, A. Anand, L. Charlin, R. D. Hjelm, P. Bachman, and A. C. Courville, “Pretraining representations for data-efficient reinforcement learning,” Advances in Neural Information Processing Systems, vol. 34, pp. 12686–12699, 2021. 





[381] M. Schwarzer, A. Anand, R. Goel, R. D. Hjelm, A. Courville, and P. Bachman, “Data-efficient reinforcement learning with self-predictive representations,” arXiv preprint arXiv:2007.05929, 2020. 





[382] D. Ha and J. Schmidhuber, “World Models,” Mar. 2018. 





[383] M. Jaderberg, V. Mnih, W. M. Czarnecki, T. Schaul, J. Z. Leibo, D. Silver, and K. Kavukcuoglu, “Reinforcement Learning with Unsupervised Auxiliary Tasks,” Nov. 2016. 





[384] I. Higgins, A. Pal, A. A. Rusu, L. Matthey, C. P. Burgess, A. Pritzel, M. Botvinick, C. Blundell, and A. Lerchner, “DARLA: Improving Zero-Shot Transfer in Reinforcement Learning,” June 2018. 





[385] C. Finn, T. Yu, J. Fu, P. Abbeel, and S. Levine, “Generalizing skills with semi-supervised reinforcement learning,” arXiv preprint arXiv:1612.00429, 2016. 





[386] R. Shah and V. Kumar, “RRL: Resnet as representation for Reinforcement Learning,” Nov. 2021. 





[387] M. Schwarzer, A. Anand, R. Goel, R. D. Hjelm, A. Courville, and P. Bachman, “Data-Efficient Reinforcement Learning with Self-Predictive Representations,” May 2021. 





[388] D. Hafner, T. Lillicrap, J. Ba, and M. Norouzi, “Dream to control: Learning behaviors by latent imagination,” arXiv preprint arXiv:1912.01603, 2019. 





[389] D. Hafner, T. Lillicrap, M. Norouzi, and J. Ba, “Mastering atari with discrete world models,” arXiv preprint arXiv:2010.02193, 2020. 





[390] F. Deng, I. Jang, and S. Ahn, “Dreamerpro: Reconstruction-free model-based reinforcement learning with prototypical representations,” in International Conference on Machine Learning, pp. 4956– 4975, PMLR, 2022. 





[391] P. Wu, A. Escontrela, D. Hafner, K. Goldberg, and P. Abbeel, “Daydreamer: World models for physical robot learning,” arXiv preprint arXiv:2206.14176, 2022. 





[392] M. Laskin, A. Srinivas, and P. Abbeel, “Curl: Contrastive unsupervised representations for reinforcement learning,” in International Conference on Machine Learning, pp. 5639–5650, PMLR, 2020. 





[393] M. Laskin, K. Lee, A. Stooke, L. Pinto, P. Abbeel, and A. Srinivas, “Reinforcement learning with augmented data,” Advances in neural information processing systems, vol. 33, pp. 19884–19895, 2020. 





[394] I. Kostrikov, D. Yarats, and R. Fergus, “Image augmentation is all you need: Regularizing deep reinforcement learning from pixels,” arXiv preprint arXiv:2004.13649, 2020. 





[395] D. Yarats, R. Fergus, A. Lazaric, and L. Pinto, “Mastering visual continuous control: Improved dataaugmented reinforcement learning,” arXiv preprint arXiv:2107.09645, 2021. 





[396] S. Nair, A. Rajeswaran, V. Kumar, C. Finn, and A. Gupta, “R3m: A universal visual representation for robot manipulation,” arXiv preprint arXiv:2203.12601, 2022. 





[397] S. Parisi, A. Rajeswaran, S. Purushwalkam, and A. Gupta, “The unsurprising effectiveness of pretrained vision models for control,” arXiv preprint arXiv:2203.03580, 2022. 





[398] C. D. Manning, P. Raghavan, and H. Schütze, Introduction to information retrieval. 2008. 





[399] R. E. Schapire and Y. Singer, “Improved boosting algorithms using confidence-rated predictions,” Mach. Learn., no. 3, 1999. 





[400] E. Reiter, “A structured review of the validity of bleu,” Computational Linguistics, 2018. 





[401] M. Denkowski and A. Lavie, “Meteor universal: Language specific translation evaluation for any target language,” in Proceedings of the ninth workshop on statistical machine translation. 





[402] C.-Y. Lin and E. Hovy, “Automatic evaluation of summaries using n-gram co-occurrence statistics,” in HLT-NAACL, 2003. 





[403] Y. Kim, “Convolutional neural networks for sentence classification,” in EMNLP. 





[404] N. Kalchbrenner, E. Grefenstette, and P. Blunsom, “A convolutional neural network for modelling sentences,” in ACL. 





[405] M. Yang, W. Zhao, J. Ye, Z. Lei, Z. Zhao, and S. Zhang, “Investigating capsule networks with dynamic routing for text classification,” in EMNLP. 





[406] L. Yao, C. Mao, and Y. Luo, “Graph convolutional networks for text classification,” in AAAI. 





[407] Y. Wang, A. Sun, J. Han, Y. Liu, and X. Zhu, “Sentiment analysis by capsules,” in WWW. 





[408] R. Socher, A. Perelygin, J. Wu, J. Chuang, C. D. Manning, A. Y. Ng, and C. Potts, “Recursive deep models for semantic compositionality over a sentiment treebank,” in EMNLP. 





[409] K. S. Tai, R. Socher, and C. D. Manning, “Improved semantic representations from tree-structured long short-term memory networks,” in ACL. 





[410] X. Zhu, P. Sobhani, and H. Guo, “Long short-term memory over recursive structures,” in ICML. 





[411] J. Cheng, L. Dong, and M. Lapata, “Long short-term memory-networks for machine reading,” in EMNLP. 





[412] P. Liu, X. Qiu, X. Chen, S. Wu, and X. Huang, “Multi-timescale long short-term memory neural network for modelling sentences and documents,” in EMNLP. 





[413] P. Liu, X. Qiu, and X. Huang, “Recurrent neural network for text classification with multi-task learning,” in IJCAI. 





[414] R. Socher, J. Pennington, E. H. Huang, A. Y. Ng, and C. D. Manning, “Semi-supervised recursive autoencoders for predicting sentiment distributions,” in EMNLP. 





[415] T. Shen, T. Zhou, G. Long, J. Jiang, and C. Zhang, “Bi-directional block self-attention for fast and memory-efficient sequence modeling,” in ICLR, 2018. 





[416] Q. V. Le and T. Mikolov, “Distributed representations of sentences and documents,” in ICML. 





[417] M. Iyyer, V. Manjunatha, J. L. Boyd-Graber, and H. D. III, “Deep unordered composition rivals syntactic methods for text classification,” in ACL. 





[418] T. Miyato, A. M. Dai, and I. J. Goodfellow, “Adversarial training methods for semi-supervised text classification,” in ICLR, 2017. 





[419] S. Lai, L. Xu, K. Liu, and J. Zhao, “Recurrent convolutional neural networks for text classification,” in AAAI. 





[420] R. Johnson and T. Zhang, “Supervised and semi-supervised text categorization using LSTM for region embeddings,” in ICML. 





[421] Y. Bao, M. Wu, S. Chang, and R. Barzilay, “Few-shot text classification with distributional signatures,” in ICLR, 2020. 





[422] F. Wu, A. H. S. Jr., T. Zhang, C. Fifty, T. Yu, and K. Q. Weinberger, “Simplifying graph convolutional networks,” in ICML. 





[423] X. Zhang, J. J. Zhao, and Y. LeCun, “Character-level convolutional networks for text classification,” in NIPS, 2015. 





[424] R. Johnson and T. Zhang, “Deep pyramid convolutional neural networks for text categorization,” in ACL. 





[425] J. Wang, Z. Wang, D. Zhang, and J. Yan, “Combining knowledge with deep convolutional neural networks for short text classification,” in IJCAI. 





[426] L. Huang, D. Ma, S. Li, X. Zhang, and H. Wang, “Text level graph neural network for text classification,” in EMNLP-IJCNLP. 





[427] C. Sun, X. Qiu, Y. Xu, and X. Huang, “How to fine-tune BERT for text classification?,” in CCL. 





[428] Z. Yang, D. Yang, C. Dyer, X. He, A. J. Smola, and E. H. Hovy, “Hierarchical attention networks for document classification,” in NAACL-HLT, 2016. 





[429] S. R. Bowman, G. Angeli, C. Potts, and C. D. Manning, “A large annotated corpus for learning natural language inference,” in EMNLP. 





[430] Z. Wang, W. Hamza, and R. Florian, “Bilateral multi-perspective matching for natural language sentences,” in IJCAI. 





[431] X. Liu, P. He, W. Chen, and J. Gao, “Multi-task deep neural networks for natural language understanding,” in ACL. 





[432] A. Williams, N. Nangia, and S. R. Bowman, “A broad-coverage challenge corpus for sentence understanding through inference,” in NAACL-HLT, 2018. 





[433] M. Marelli, L. Bentivogli, M. Baroni, R. Bernardi, S. Menini, and R. Zamparelli, “Semeval-2014 task 1: Evaluation of compositional distributional semantic models on full sentences through semantic relatedness and textual entailment,” in SemEval@COLING. 





[434] B. Dolan, C. Quirk, and C. Brockett, “Unsupervised construction of large paraphrase corpora: Exploiting massively parallel news sources,” in COLING, 2004. 





[435] J. Fu, P. Liu, and G. Neubig, “Interpretable multi-dataset evaluation for named entity recognition,” in EMNLP. 





[436] B. Lester, D. Pressel, A. Hemmeter, S. R. Choudhury, and S. Bangalore, “Constrained decoding for computationally efficient named entity recognition taggers,” in EMNLP. 





[437] Y. Luo, H. Zhao, and J. Zhan, “Named entity recognition only from word embeddings,” in EMNLP. 





[438] X. Li, J. Feng, Y. Meng, Q. Han, F. Wu, and J. Li, “A unified MRC framework for named entity recognition,” in ACL. 





[439] Y. Zhang and J. Yang, “Chinese NER using lattice LSTM,” in ACL. 





[440] Y. Meng, W. Wu, F. Wang, X. Li, P. Nie, F. Yin, M. Li, Q. Han, X. Sun, and J. Li, “Glyce: Glyphvectors for chinese character representations,” in NeurIPS. 





[441] A. Katiyar and C. Cardie, “Nested named entity recognition revisited,” in NAACL-HLT, 2018. 





[442] B. Wang and W. Lu, “Neural segmental hypergraphs for overlapping mention recognition,” in EMNLP. 





[443] Y. Luan, D. Wadden, L. He, A. Shah, M. Ostendorf, and H. Hajishirzi, “A general framework for information extraction using dynamic span graphs,” in NAACL-HLT, 2019. 





[444] T. Shibuya and E. H. Hovy, “Nested named entity recognition via second-best sequence learning and decoding,” Trans. Assoc. Comput. Linguistics, 2020. 





[445] H. Lin, Y. Lu, X. Han, and L. Sun, “Sequence-to-nuggets: Nested entity mention detection via anchorregion networks,” in ACL. 





[446] G. Lai, Q. Xie, H. Liu, Y. Yang, and E. H. Hovy, “RACE: large-scale reading comprehension dataset from examinations,” in EMNLP. 





[447] Y. Yang, W. Yih, and C. Meek, “Wikiqa: A challenge dataset for open-domain question answering,” in EMNLP. 





[448] C. N. dos Santos, M. Tan, B. Xiang, and B. Zhou, “Attentive pooling networks,” CoRR, 2016. 





[449] J. Y. Lee and F. Dernoncourt, “Sequential short-text classification with recurrent and convolutional neural networks,” in NAACL-HLT, 2016. 





[450] S. Kim, L. F. D’Haro, R. E. Banchs, J. D. Williams, and M. Henderson, “The fourth dialog state tracking challenge,” in Dialogues with Social Robots - Enablements, Analyses, and Evaluation, Seventh International Workshop on Spoken Dialogue Systems, IWSDS 2016, Saariselkä, Finland, January 13-16, 2016, 2016. 





[451] J. Ang, Y. Liu, and E. Shriberg, “Automatic dialog act segmentation and classification in multiparty meetings,” in 2005 IEEE International Conference on Acoustics, Speech, and Signal Processing, ICASSP ’05, Philadelphia, Pennsylvania, USA, March 18-23, 2005, 2005. 





[452] Y. Wan, W. Yan, J. Gao, Z. Zhao, J. Wu, and P. S. Yu, “Improved dynamic memory network for dialogue act classification with adversarial training,” in IEEE International Conference on Big Data, Big Data 2018, Seattle, WA, USA, December 10-13, 2018, 2018. 





[453] V. Raheja and J. R. Tetreault, “Dialogue act classification with context-aware self-attention,” in Proc. NAACL, 2019, 2019. 





[454] J. Xu, Z. Gan, Y. Cheng, and J. Liu, “Discourse-aware neural extractive text summarization,” in ACL. 





[455] Y. Zou, X. Zhang, W. Lu, F. Wei, and M. Zhou, “Pre-training for abstractive document summarization by reinstating source text,” in EMNLP. 





[456] L. Liu, Y. Lu, M. Yang, Q. Qu, J. Zhu, and H. Li, “Generative adversarial network for abstractive text summarization,” in AAAI. 





[457] M. Yang, Q. Qu, W. Tu, Y. Shen, Z. Zhao, and X. Chen, “Exploring human-like reading strategy for abstractive text summarization,” in AAAI. 





[458] M. Bhandari, P. N. Gour, A. Ashfaq, P. Liu, and G. Neubig, “Re-evaluating evaluation in text summarization,” in EMNLP. 





[459] Y. Dong, S. Wang, Z. Gan, Y. Cheng, J. C. K. Cheung, and J. Liu, “Multi-fact correction in abstractive text summarization,” in EMNLP. 





[460] D. Huang, L. Cui, S. Yang, G. Bao, K. Wang, J. Xie, and Y. Zhang, “What have we achieved on text summarization?,” in EMNLP. 





[461] W. Kryscinski, R. Paulus, C. Xiong, and R. Socher, “Improving abstraction in text summarization,” in EMNLP. 





[462] W. Kryscinski, B. McCann, C. Xiong, and R. Socher, “Evaluating the factual consistency of abstractive text summarization,” in EMNLP. 





[463] P. Kouris, G. Alexandridis, and A. Stafylopatis, “Abstractive text summarization based on deep learning and semantic content generalization,” in ACL. 





[464] K. Chen, R. Wang, M. Utiyama, and E. Sumita, “Content word aware neural machine translation,” in ACL, 2020. 





[465] Z. Lin, X. Pan, M. Wang, X. Qiu, J. Feng, H. Zhou, and L. Li, “Pre-training multilingual neural machine translation by leveraging alignment information,” in EMNLP. 





[466] E. Bugliarello and N. Okazaki, “Enhancing machine translation with dependency-aware selfattention,” in ACL, 2020. 





[467] A. F. Aji, N. Bogoychev, K. Heafield, and R. Sennrich, “In neural machine translation, what does transfer learning transfer?,” in ACL. 





[468] C. Baziotis, B. Haddow, and A. Birch, “Language model prior for low-resource neural machine translation,” in EMNLP. 





[469] Q. Cui, S. Huang, J. Li, X. Geng, Z. Zheng, G. Huang, and J. Chen, “Directqe: Direct pretraining for machine translation quality estimation,” in AAAI. 





[470] C. Wu, S. C. H. Hoi, R. Socher, and C. Xiong, “TOD-BERT: pre-trained natural language understanding for task-oriented dialogue,” in EMNLP. 





[471] G. Campagna, A. Foryciarz, M. Moradshahi, and M. S. Lam, “Zero-shot transfer learning with synthesized data for multi-domain dialogue state tracking,” in ACL, 2020. 





[472] Q. Liu, L. Yu, L. Rimell, and P. Blunsom, “Pretraining the noisy channel model for task-oriented dialogue,” CoRR, 2021. 





[473] “SST Corpus.” http://nlp.stanford.edu/sentiment, 2013. 





[474] B. Pang and L. Lee, “Seeing stars: Exploiting class relationships for sentiment categorization with respect to rating scales,” in ACL. 





[475] R. Socher, A. Perelygin, J. Wu, J. Chuang, C. D. Manning, A. Y. Ng, and C. Potts, “Recursive deep models for semantic compositionality over a sentiment treebank,” in EMNLP. 





[476] D. Cer, M. Diab, E. Agirre, I. Lopez-Gazpio, and L. Specia, “Semeval-2017 task 1: Semantic textual similarity-multilingual and cross-lingual focused evaluation,” arXiv, 2017. 





[477] I. Hendrickx, S. N. Kim, Z. Kozareva, P. Nakov, D. Ó. Séaghdha, S. Padó, M. Pennacchiotti, L. Romano, and S. Szpakowicz, “Semeval-2010 task 8: Multi-way classification of semantic relations between pairs of nominals,” in Proc. NAACL, 2009, 2009. 





[478] J. Wiebe, T. Wilson, and C. Cardie, “Annotating expressions of opinions and emotions in language,” Language Resources and Evaluation, no. 2-3, 2005. 





[479] “MPQA Corpus.” http://www.cs.pitt.edu/mpqa/, 2005. 





[480] Q. Diao, M. Qiu, C. Wu, A. J. Smola, J. Jiang, and C. Wang, “Jointly modeling aspects, ratings and sentiments for movie recommendation (JMARS),” in ACM SIGKDD. 





[481] “20NG Corpus.” http://ana.cachopo.org/datasets-for-single-label-text-categorizati 2007. 





[482] “AG Corpus.” http://www.di.unipi.it/~gulli/AG_corpus_of_news_articles. html, 2004. 





[483] “Reuters Corpus.” https://www.cs.umb.edu/~smimarog/textmining/datasets/, 2007. 





[484] “Reuters Corpus.” https://martin-thoma.com/nlp-reuters, 2017. 





[485] J. Lehmann, R. Isele, M. Jakob, A. Jentzsch, D. Kontokostas, P. N. Mendes, S. Hellmann, M. Morsey, P. van Kleef, S. Auer, and C. Bizer, “Dbpedia - A large-scale, multilingual knowledge base extracted from wikipedia,” Semantic Web, no. 2, 2015. 





[486] “Ohsumed Corpus.” http://davis.wpi.edu/xmdv/datasets/ohsumed.html, 2015. 





[487] A. Williams, N. Nangia, and S. R. Bowman, “A broad-coverage challenge corpus for sentence understanding through inference,” arXiv, 2017. 





[488] P. Rajpurkar, J. Zhang, K. Lopyrev, and P. Liang, “Squad: $1 0 0 { , } 0 0 0 { + }$ questions for machine comprehension of text,” arXiv, 2016. 





[489] H. Levesque, E. Davis, and L. Morgenstern, “The winograd schema challenge,” in Thirteenth International Conference on the Principles of Knowledge Representation and Reasoning, 2012. 





[490] W. B. Dolan and C. Brockett, “Automatically constructing a corpus of sentential paraphrases,” in IWP, 2005. 





[491] P. Rajpurkar, R. Jia, and P. Liang, “Know what you don’t know: Unanswerable questions for squad,” arXiv, 2018. 





[492] G. Lai, Q. Xie, H. Liu, Y. Yang, and E. Hovy, “Race: Large-scale reading comprehension dataset from examinations,” arXiv, 2017. 





[493] D. Jurafsky and E. Shriberg, “Switchboard swbd-damsl shallow-discourse-function annotation coders manual,” 1997. 





[494] J. Li, P. Zhou, C. Xiong, R. Socher, and S. C. Hoi, “Prototypical contrastive learning of unsupervised representations,” arXiv preprint arXiv:2005.04966, 2020. 





[495] J. Donahue and K. Simonyan, “Large scale adversarial representation learning,” Advances in Neural Information Processing Systems, vol. 32, 2019. 





[496] K. He, X. Chen, S. Xie, Y. Li, P. Dollár, and R. Girshick, “Masked autoencoders are scalable vision learners,” arXiv preprint arXiv:2111.06377, 2021. 





[497] http://yann.lecun.com/exdb/mnist/. 





[498] http://ufldl.stanford.edu/housenumbers/. 





[499] https://www.cs.toronto.edu/~kriz/index.html. 





[500] A. Coates, A. Ng, and H. Lee, “An analysis of single-layer networks in unsupervised feature learning,” in Proceedings of the fourteenth international conference on artificial intelligence and statistics, 2011. 





[501] https://cs.stanford.edu/~acoates/stl10/. 





[502] http://www.vision.caltech.edu/Image_Datasets/Caltech101/. 





[503] G. A. Miller, WordNet: An electronic lexical database. 1998. 





[504] https://image-net.org/. 





[505] https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/. 





[506] H. Kuehne, H. Jhuang, E. Garrote, T. Poggio, and T. Serre, “HMDB: a large video database for human motion recognition,” in ICCV, 2011. 





[507] https://www.crcv.ucf.edu/data/UCF101.php. 





[508] https://www.crcv.ucf.edu/data/UCF50.php. 





[509] L. Bossard, M. Guillaumin, and L. Van Gool, “Food-101–mining discriminative components with random forests,” in European conference on computer vision, 2014. 





[510] T. Berg, J. Liu, S. Woo Lee, M. L. Alexander, D. W. Jacobs, and P. N. Belhumeur, “Birdsnap: Largescale fine-grained visual categorization of birds,” in CVPR, 2014. 





[511] J. Xiao, J. Hays, K. A. Ehinger, A. Oliva, and A. Torralba, “Sun database: Large-scale scene recognition from abbey to zoo,” in 2010 IEEE computer society conference on computer vision and pattern recognition, 2010. 





[512] J. Xiao, K. A. Ehinger, J. Hays, A. Torralba, and A. Oliva, “Sun database: Exploring a large collection of scene categories,” International Journal of Computer Vision, 2016. 





[513] http://places.csail.mit.edu/downloadData.html. 





[514] http://ai.stanford.edu/~jkrause/cars/car_dataset.html. 





[515] S. Maji, J. Kannala, E. Rahtu, M. Blaschko, and A. Vedaldi, “Fine-grained visual classification of aircraft,” tech. rep., 2013. 





[516] https://sites.google.com/site/fgcomp2013/. 





[517] https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/. 





[518] https://www.robots.ox.ac.uk/~vgg/data/pets/. 





[519] https://www.robots.ox.ac.uk/~vgg/data/flowers/. 





[520] https://www.robots.ox.ac.uk/~vgg/data/dtd/. 





[521] https://sites.google.com/view/fgvc5/competitions/inaturalist. 





[522] https://www.inaturalist.org/. 





[523] C. Sun, A. Shrivastava, S. Singh, and A. Gupta, “Revisiting unreasonable effectiveness of data in deep learning era,” in Proceedings of the IEEE international conference on computer vision, pp. 843–852, 2017. 





[524] http://host.robots.ox.ac.uk/pascal/VOC/. 





[525] http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html. 





[526] http://host.robots.ox.ac.uk/pascal/VOC/voc2011/index.html. 





[527] http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html. 





[528] B. Zhou, H. Zhao, X. Puig, S. Fidler, A. Barriuso, and A. Torralba, “Scene parsing through ade20k dataset,” in CVPR. 





[529] B. Zhou, H. Zhao, X. Puig, T. Xiao, S. Fidler, A. Barriuso, and A. Torralba, “Semantic understanding of scenes through the ade20k dataset,” International Journal of Computer Vision, 2019. 





[530] https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html. 





[531] M. Cordts, M. Omran, S. Ramos, T. Scharwächter, M. Enzweiler, R. Benenson, U. Franke, S. Roth, and B. Schiele, “The cityscapes dataset,” in CVPR Workshop on The Future of Datasets in Vision, 2015. 





[532] M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler, R. Benenson, U. Franke, S. Roth, and B. Schiele, “The cityscapes dataset for semantic urban scene understanding,” in Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016. 





[533] A. Gupta, P. Dollar, and R. Girshick, “LVIS: A dataset for large vocabulary instance segmentation,” in CVPR, 2019. 





[534] https://davischallenge.org/. 





[535] https://davischallenge.org/davis2017/code.html. 





[536] C. Doersch, “Data analysis project: What makes paris look like paris?,” 





[537] http://www.cs.toronto.edu/~nitish/unsupervised_video/. 





[538] N. Srivastava, E. Mansimov, and R. Salakhudinov, “Unsupervised learning of video representations using lstms,” in International conference on machine learning, 2015. 





[539] B. Thomee, D. A. Shamma, G. Friedland, B. Elizalde, K. Ni, D. Poland, D. Borth, and L.-J. Li, “Yfcc100m: The new data in multimedia research,” Communications of the ACM, 2016. 





[540] http://projects.dfki.uni-kl.de/yfcc100m/. 





[541] W. Jin, X. Liu, X. Zhao, Y. Ma, N. Shah, and J. Tang, “Automated self-supervised learning for graphs,” CoRR, 2021. 





[542] Z. Peng, Y. Dong, M. Luo, X. Wu, and Q. Zheng, “Self-supervised graph representation learning via global context prediction,” CoRR, 2020. 





[543] Y. Zhu, Y. Xu, F. Yu, Q. Liu, S. Wu, and L. Wang, “Deep graph contrastive representation learning,” CoRR, 2020. 





[544] M. Jin, Y. Zheng, Y. Li, C. Gong, C. Zhou, and S. Pan, “Multi-scale contrastive siamese networks for self-supervised graph representation learning,” CoRR, 2021. 





[545] Z. Hu, C. Fan, T. Chen, K. Chang, and Y. Sun, “Pre-training graph neural networks for generic structural feature extraction,” CoRR, 2019. 





[546] Y. Zhu, Y. Xu, F. Yu, S. Wu, and L. Wang, “CAGNN: cluster-aware graph neural networks for unsupervised graph representation learning,” CoRR, 2020. 





[547] H. Zhang, S. Lin, W. Liu, P. Zhou, J. Tang, X. Liang, and E. P. Xing, “Iterative graph self-distillation,” CoRR, 2020. 





[548] S. Lin, P. Zhou, Z.-Y. Hu, S. Wang, R. Zhao, Y. Zheng, L. Lin, E. Xing, and X. Liang, “Prototypical graph contrastive learning,” 2021. 





[549] S. Zhang, Z. Hu, A. Subramonian, and Y. Sun, “Motif-driven contrastive learning of graph representations,” CoRR, 2020. 





[550] F. L. Opolka, A. Solomon, C. Cangea, P. Velickovic, P. Liò, and R. D. Hjelm, “Spatio-temporal deep graph infomax,” CoRR, 2019.