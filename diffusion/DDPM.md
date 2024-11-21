生成扩散模型漫谈（一）：DDPM = 拆楼 + 建楼

(12)的推导
```math
\left\{\begin{aligned} 
\alpha_t\bar{\beta}_{t-1}\bar{\boldsymbol{\varepsilon}}_{t-1} + \beta_t \boldsymbol{\varepsilon}_t = \bar{\beta}_t\boldsymbol{\varepsilon}\\ 
\beta_t \bar{\boldsymbol{\varepsilon}}_{t-1} - \alpha_t\bar{\beta}_{t-1} \boldsymbol{\varepsilon}_t = \bar{\beta}_t\boldsymbol{\omega} 
\end{aligned}\right.
```
```math
\left\{\begin{aligned} 
\beta_t^2 \boldsymbol{\varepsilon}_t - (\alpha_t\bar{\beta}_{t-1} \boldsymbol)^2{\varepsilon}_t = \bar\beta_t{\beta}_t\boldsymbol{\varepsilon} - \bar\alpha_t\bar{\beta}_{t-1} \boldsymbol{\beta}_t\boldsymbol{\omega}\\ 
\end{aligned}\right.
```
```math
\begin{equation}\boldsymbol{\varepsilon}_t = \frac{(\beta_t \boldsymbol{\varepsilon} - \alpha_t\bar{\beta}_{t-1} \boldsymbol{\omega})\bar{\beta}_t}{\beta_t^2 + \alpha_t^2\bar{\beta}_{t-1}^2} = \frac{\beta_t \boldsymbol{\varepsilon} - \alpha_t\bar{\beta}_{t-1} \boldsymbol{\omega}}{\bar{\beta}_t}\end{equation}
```
[https://kexue.fm/archives/9119/comment-page-1#comment-19316](https://kexue.fm/archives/9119/comment-page-1#comment-19316)
