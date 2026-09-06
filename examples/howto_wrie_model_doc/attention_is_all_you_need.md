$$
\begin{gather*}
    N\text{: batch} \\
    S\text{: длина входной последовательности} \\
    d_{model}\text{: размер вектора из входной и выходной последовательности} \\
    T\text{: длина выходной последовательности} \\
    d_v\text{: value embedding dim, size of vectors in projection space in each layer} \\
    d_k\text{: query\&key embedding dim, size of vectors in projection space in each layer}\\
    d_{model}\text{: size of vectors flowing between layers}
\end{gather*}
$$
___

$$
\text{context}: (N,S,d_{model})
$$
$$
\text{horizon}: (N,T,d_{model})
$$


$$
\begin{gather*}
\text{context}^{N \times S \times d_{model}}\\
\text{horizon}^{N \times T \times d_{model}}\\
\end{gather*}
$$

___

# ATTENTION  

$$
    \text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1,\dots,\text{head}_h)W^O
$$

$$
\text{head}(Q,K,V) = \text{Attention}(QW^Q, KW^K, VW^V)
$$

$$
\begin{gather*}
\text{Attention} = \text{softmax}(\frac{QK^{T}}{\sqrt{d_k}})V\\ 
\text{Attention}:\text{softmax}(N \times T \times S) N \times S \times d_{model} \rightarrow N \times T \times d_{model}
\end{gather*}
$$


$$
Q \in \mathbb{R}^{N \times T  \times d_{model}}
$$

$$
K \in \mathbb{R}^{N \times S  \times d_{model}}
$$

$$
V \in \mathbb{R}^{N \times S  \times d_{model}}
$$

$$
W^{Q} \in \mathbb{R}^{d_{model} \times d_k}
$$

$$
W^{K} \in \mathbb{R}^{d_{model} \times d_k}
$$

$$
W^{V} \in \mathbb{R}^{d_{model} \times d_v}
$$

$$
W^{O} \in \mathbb{R}^{hd_v \times d_{model}}
$$

$$
d_k = d_v = \frac{d_{model}}{h}
$$

$$
\begin{gather*}
\text{projection:} \\
Q^{N \times T \times d_{model}} W^{d_{model} \times d_k} \rightarrow Q_{proj}^{N \times T \times d_k}
\end{gather*}
$$

$$
\begin{gather*}
\text{attention head with linear projection}:\\
  \big[N \times T  \times d_k \text{ MATMUL }  N \times d_k \times S \big] \text{ MATMUL } N \times S \times d_v \rightarrow 
  N \times T \times d_v  
\end{gather*}
$$

$$
\begin{gather*}
\text{multi head attention}:\\
  \text{ CONCAT 2-dim } [\text{ REPEAT } h \text{ TIMES }  N \times T \times d_v] \text{ MATMUL } hd_v \times d_{model}\rightarrow 
  N \times T \times h d_v \text{ MATMUL } hd_v \times d_{model} 
  \rightarrow N \times T \times d_{model}
\end{gather*}
$$


# SELF ATTENTION  

$$
\text{self\_attention}(\text{seq}) = \text{Attention}(Q=\text{seq},K=\text{seq},V=\text{seq})
$$

# ATTENTION WITH MASK  

$$
\text{key\_padding\_mask} \in \mathbb{R}^{N \times S}
$$  

$$
\text{attention\_mask} \in \mathbb{R}^{ N \times T \times S}
$$



$$
\begin{gather*}
\text{softmax}(
\frac{1}{\sqrt{d_k}}
\begin{bmatrix}
Q_1 \\
Q_2 \\
\vdots \\
Q_T \
\end{bmatrix}
\begin{bmatrix}
K_1 & K_2 & \dots K_S
\end{bmatrix}
)
\begin{bmatrix}
V_1 \\
V_2 \\
\vdots \\
V_S \
\end{bmatrix} 
\rightarrow
\text{softmax}(
  \frac{1}{\sqrt{d_k}}
\begin{bmatrix}
Q_1 K_1 & Q_1 K_2 & \dots & Q_1 K_S \\
Q_2 K_1 & Q_2 K_2 & \dots & Q_2 K_S \\
\vdots & \vdots & \vdots & \vdots \\
Q_T K_1 & Q_T K_2 &\dots & Q_T K_S \\
\end{bmatrix}
)
\begin{bmatrix}
V_1 \\
V_2 \\
\vdots \\
V_S \
\end{bmatrix}\\
\end{gather*}
$$

$$
\begin{gather*}
Q\in \mathbb{R}^{N \times h \times T \times d_{k}}\\ 
K\in \mathbb{R}^{N \times h \times S \times d_{k}}\\
V\in \mathbb{R}^{N \times h \times S \times d_{k}}
\end{gather*}
$$

$$
\text{mask with inf}_{ij} =
\begin{cases}
        0 \text{ , $i \geq j $}
        \\
        - \infty \text{ , $ i < j $}
\end{cases}
$$

$$
\text{binary mask}_{ij} =
\begin{cases}
        1 \text{ , $i \geq j $}
        \\
        0 \text{ , $ i < j $}
\end{cases}
$$

$$
\begin{gather*}
\text{attn\_scores} = \frac{1}{\sqrt{d_k}}
\begin{bmatrix}
Q_1 K_1 & Q_1 K_2 & \dots & Q_1 K_S \\
Q_2 K_1 & Q_2 K_2 & \dots & Q_2 K_S \\
\vdots & \vdots & \vdots & \vdots \\
Q_T K_1 & Q_T K_2 &\dots & Q_T K_S \\
\end{bmatrix}\\
\text{attn\_weights}=\text{relu}(\text{attn\_scores} \odot \text{binary mask})  \text{ \# if relu }\\
\text{attn\_weights}=\text{softmax}(\text{attn\_scores} + \text{mask with }-\infty) \text{ \# if softmax or relu}\\
\text{attn\_weights} = \text{attn\_weights} \odot \text{drop out mask \# if noise}\\
\text{attn\_output} = \text{attn\_weights} \text{ MATMUL } V
\end{gather*}
$$




$$
\text{multi\_head\_attention\_mask} \in \mathbb{R}^{ N \cdot h \times T \times S}
$$

# ATTENTION WITH MEMORY
$$
\text{attention}(\text{seq}, \text{memory}) = \text{Attention}(Q=\text{seq},K=\text{memory},V=\text{memory})
$$




# FORWARD PASS  

$$
\begin{gather*}
\text{memory} = \text{encoder}(\text{context})\\
\text{horizon} = \text{decoder}(\text{memory},\text{tgt})
\end{gather*}
$$


# ENCODER IMPL  

$$
\text{feed\_forward} = \text{linear}_2 \circ \text{dropout} \circ \text{activation} \circ \text{linear}_1
$$

layer_norm = https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html


$$
\begin{gather*}
A_{1} = \text{layer\_norm} \circ (\text{self\_attention} + I) \\
A_{2} = \text{layer\_norm} \circ (\text{feed\_forward} + I)
\end{gather*}
$$

$$
\text{encoder\_layer} = A_{2} \circ A_{1}
$$

$$
\text{encoder} =   \text{encoder\_layer}_{n} \circ \dots \circ \text{encoder\_layer}_{1}
$$

$$
\text{memory} = \text{encoder}(\text{context})
$$


# DECODER IMPL  

$$
\begin{gather*}
A_{1} = \text{layer\_norm} \circ (\text{self\_attention} + I)\\ 
A_{2}(\cdot,\text{memory}) = \text{layer\_norm} \circ (\text{multi\_head}(\cdot,\text{memory})+I)\\
A_{3} = \text{layer\_norm} \circ (\text{feed\_forward}+I)
\end{gather*}
$$

$$
\begin{gather*}
\text{decoder\_layer}(\cdot,\text{memory})= A_{3} \circ A_{2}(\cdot,\text{memory}) \circ A_{1}\\
\text{decoder\_layer}(\text{memory,tgt}) = A_{3}(A_{2}(\text{memory}, A_{1}(\text{tgt})))
\end{gather*} 
$$


$$
\text{decoder}(\cdot,\text{memory}) = \text{decoder\_layer}_{n}(\cdot,\text{memory}) \circ \dots \circ \text{decoder\_layer}_{1}(\cdot,\text{memory})
$$

$$
\text{decoder}(\text{tgt},\text{memory}) = \text{decoder\_layer}_{n}(\dots(\text{decoder\_layer}_{2}(\text{decoder\_layer}_{1}(\text{tgt},\text{memory}),\text{memory})\dots),\text{memory})
$$


# Teacher Forcing  

$$
\begin{gather*}
\text{decoder input: } [\text{<sos>}, Y_{1},Y_{2},...,Y_{horizon-1},\text{<pad>},\text{<eos>}] \cup \text{memory}\\ 
\text{decoder output: } [Y_{1},Y_{2},...,Y_{horizon},\text{<eos>}] 
\end{gather*}
$$

# Padding for fixed shapes on inference stage
first step:  
$$
\begin{gather*}
\text{decoder input: } [\text{<sos>},\text{<pad>}\dots,\text{<pad>}] \cup \text{memory}\\ 
\text{decoder output: } [Y_{1},\text{<pad>}\dots,\text{<pad>}] 
\end{gather*}
$$

t-th step:  
$$
\begin{gather*}
\text{decoder input: } [\text{<sos>}, Y_{1},\dots,Y_{t},\text{<pad>}\dots,\text{<pad>}] \cup \text{memory}\\ 
\text{decoder output: } [Y_{1},\dots,Y_{t+1},\text{<pad>}\dots,\text{<pad>}] 
\end{gather*}
$$

$$
\text{mask with inf}_{ij} =
\begin{cases}
        0 \text{ , $j \leq t$}
        \\
        - \infty \text{ , $ j > t $}
\end{cases}
$$

$$
\text{binary mask}_{ij} =
\begin{cases}
        1 \text{ , $j \leq t$}
        \\
        0 \text{ , $ j > t $}
\end{cases}
$$

