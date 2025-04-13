# 7404_Reproduction-of-Efficient-Classification-for-Additive-Kernel-SVMs
In this project, we reproduce the key findings of the paper \textit{Efficient Classification for Additive Kernel SVMs}, with a focus on evaluating the effectiveness and acceleration benefits of the proposed method. The original work addresses the computational bottleneck of nonlinear SVMs by reformulating the decision function to reduce inference complexity from $O(mn)$ to $O(2(k+1)n)$. Through a series of experiments on histogram-based vision datasets, we demonstrate that IKSVM significantly improves efficiency while maintaining competitive accuracy. Furthermore, we implement and compare the Nystroem approximation method. Although Nystroem provides speed-up and performs comparably in some cases, IKSVM consistently outperforms it in both classification accuracy and runtime when applied to additive kernels. 
