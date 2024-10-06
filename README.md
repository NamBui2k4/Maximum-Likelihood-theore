Bài viết này được kham khảo từ nguồn: 

- [_phamdinhkhanh.github_](https://phamdinhkhanh.github.io/deepai-book/ch_ml/NaiveBayes.html)

- [_kierandg.blogspot.com_](https://kierandg.blogspot.com/2016/05/so-luoc-log-likelihood-maximum-likelihood-estimation.html)

- [_dangnguyenit.blogspot.com_](https://dangnguyenit.blogspot.com/2018/10/uoc-luong-hop-ly-cuc-aimaximum.html)

# Ước lượng khả năng tối đa (Maximum Likelihood Estimation - MLE)

Trong thống kê và nghiên cứu khoa học nói chúng, dữ liệu thường được diễn tả thông qua những __phân phối xác suất__. Phân phối xác suất thường được đặc trưng bởi những tham số nhất định. Ví dụ, đối với phân phối chuẩn, tham số đặc trưng chính là trung bình ($μ $) và phương sai ($σ^2$). Đối với phân phối Poisson 
thì tham số đặc trưng là tỷ lệ trung bình λ.

Tuy nhiên, có những tình huống khó khăn khi chúng ta không thể đưa ra những con số thống kê tuyệt đối này trên một tổng thể lớn. Chẳng hạn, bạn muốn
thống kê chiều cao trung bình của người Việt Nam nhưng lại không thể đi khắp cái nước Việt Nam 100 triệu người này và hỏi người ta "bạn cao bao nhiêu m vậy". 
Bạn chỉ có thể đi hỏi một vài người và biết rằng chiều cao trung bình là 1m62, nhưng lại không chắc chắn con số này có đúng với 100 triệu người không.
Đó là lúc mà giải pháp **ước lượng khả năng tối đa** (Maximum Likelihood Estimation - viết tắt là MLE) ra đời. 

Tuy nhiên, trước hết, ta cần hiểu rõ __hàm khả năng(Likelihood function)__ là gì.

## Định nghĩa hàm khả năng

    Trong thống kê, Likelihood function - gọi là
    hàm khả năng, được hiểu là một hàm đo lường
    mức độ "phù hợp" của một bộ tham số cụ thể với
    dữ liệu quan sát được. 
\
Giả sử biến ngẫu nhiên  X  tuân theo một phân phối nào đó được mô tả bởi bộ tham số $θ ( θ_1,θ_2,...,θ_k )$ mà ta chưa biết.

Hàm khả năng của phân phối có dạng:

- $L(θ)= f(x_1,x_2,...,x_n | θ_1,θ_2,...,θ_k)$ (nếu  X  là biến liên tục)
- $L(θ)= p(x_1,x_2,...,x_n | θ_1,θ_2,...,θ_k)$ (nếu  X  là biến rời rạc)

Trong đó:
- $x_1,x_2,...,x_n$ là các giá trị mà X có thể nhận trong mẫu dữ liệu
- $θ_1,θ_2,...,θ_k$ là tập hợp tham số của phân phối dữ liệu

Hàm khả năng có thể được hiểu là xác suất để các sự kiện  $x_1,x_2,...,x_n$ cùng xảy ra, với điều kiện $( θ_1,θ_2,...,θ_k )$. 
Như vậy, cách gọi "hàm khả năng" ở đây chính là xác suất có điều kiện. 

## Định nghĩa ước lượng khả năng tối đa (Maximum Likelihood definition)

    Trong thống kê, ước lượng khả năng tối đa là một
    phương pháp ước lượng tham số của phân phối dữ
    liệu bằng cách tối đa hoá hàm khả năng sao cho
    dưới giả định của thống kê thì dữ liệu
    trở nên phù hợp nhất.
\

Nói đơn giản, ước lượng khả năng tối đa là tìm θ để $L(θ)$ đạt max.

$$θ = argmax{L(θ)}$$

Tưởng đơn giản ấy, nhưng mà không nhé. Vì ta phải giải quyết bài toán "tính xác suất để X nhận được tất cả các giá trị $x_1,x_2,...,x_n$". Mặt khác, ta đã nói rằng  hàm likelihood bản chất là xác suất có điều kiện. Khi đó, nếu $X$ liên tục, ta biến đổi như sau:

 $$L(θ) = f(x_1,x_2,...,x_n | θ) = f(x_1|θ).f(x_2|θ)...f(x_n|θ)$$

Nếu X rời rạc, ta biến đổi:

$$L(θ)= p(x_1,x_2,...,x_n | θ_1,θ_2,...,θ_k) = p(x_1|θ) + p(x_2|θ) +...+ p(x_n|θ_k)$$

(Nói chung thì cũng chỉ là khác nhau giữa dấu nhân và dấu cộng :v )

Việc tính toán likelihood sẽ gặp rất nhiều trở ngại. Đấy là chưa kể ta không biết θ sẽ chứa cái gì (tùy thuộc vào loại phân phối mà ta giả định thì
θ có thể chứa μ, σ, λ,...). Lúc này bài toán có 2 kiểu: Ước lượng một tham số và Ước lượng nhiều tham số 

## Ước lượng một tham số

Bài toán lúc này yêu cầu chúng ta ước lượng likelihood cho một tham số trong bộ θ, chẳng hạn như ước lượng tham số p trong phân phối Bernoulli(p), hay tình huống phân phối chuẩn N(μ,σ) đã cho trước 1 trong 2 giá trị μ,σ và yêu cầu ta ước lượng cho cái còn lại. Khi đó, chúng ta thực hiện cái bước:

1. Xác định likelihood
2. Biến đổi likelihood về dạng logarit: $u(θ) = log(L(θ))$.  
3. Đạo hàm riêng, ta được : $u(θ)' = log(L(θ))'$
4. likelihood đạt cực đại
   
     ⇔ $u(θ)$đạt cực đại
 
     ⟺ $u(θ)' = 0$
 
     ⟺ tham số trong $θ$ bằng một giá trị nào đó
5. kết luận $θ$

__Ví dụ__

Cho  $x_1,x_2,...,x_n$∼  Bernoulli(p). Dùng phương pháp Ước lượng khả năng cực đại(MLE) để ước lượng tham số p .

_Giải:_

Phân phối Bernoulli(p) là phân phối xác suất của biến liên tục x với phân phối xác suất là: $f(x|θ) = p^{x}(1−p)^{1−x}  $


1. Xác định Likelihood: 
$$L(p) = f(x_1,x_2,...,x_n | p) = f(x_1|p).f(x_2|p)...f(x_n|p)$$

$$=∏_{i = 1}^{n} p^{x_i}(1−p)^{1−x_i}$$

2. Dùng logarit đưa likelihood về dạng tổng

Đặt $u(p) = log(L(p)) $, ta có:

$$u(p) = log(∏_{i = 1}^{n} p^{x_i}(1−p)^{1−x_i} )$$

$$= ∑_{i = 1}^{n} log(p^{x_i}(1−p)^{1−x_i})$$

$$= ∑_{i=1}^{n} log(p^{x_i})+ ∑_{i=1}^{n}log(1−p)^{1−x_i}$$

$$= ∑_{i=1}^{n} x_i \ log(p) + ∑_{i=1}^{n}(1−x_i) \ log(1−p)$$


Đặt  $∑_{i=1}^{n} x_i = t$, ta coi t như một hằng số thì phương trình trở thành:

$u(p) = t  \  log(p) + (n - t)  \  log(1-p)$

3. đạo hàm: $u'(p) = t \ \frac{1}{p} +(n−t) \  \frac{−1}{1−p}$

4. likelihood đạt cực đại
   
⟺ $u(p)$ đạt cực đại

⟺ $u'(p) = 0 $

⟺ $t \ \frac{1}{p} +(n−t) \  \frac{−1}{1−p}=0$

⟺ $p = \frac{1}{n} ∑_{i=1}^{n}x_i$

6. Kết luận:  $p = \frac{1}{n} ∑_{i=1}^{n}x_i$ là ước lượng khả năng tối đa cần tìm

Nếu ta để ý thì  $p = \frac{1}{n} ∑_{i=1}^{n}x_i$ chính là trung bình mẫu $\bar{x}$ của mẫu của mẫu dữ liệu mà ta thu thập ban đầu.

__Bảng tóm tắt công thức ước lượng khả năng tối đa của một số phân phối__

| Phân phối      |              MLE           |
|----------------|----------------------------|
| Bernoulli(p)   | $p = \bar{x} $           |
| Normal (μ, $σ^2$)  | $μ = \bar{x}$            |
| Exp(λ)         | $λ = \bar{x}^{-1}$        |
| Geometric(p)   | $p = \frac{1}{\bar{x}}  $|
| Binominal(p,n) | $p = \frac{\bar{x}}{n}$  |
| Poisson(λ)     | $λ = \bar{x}$            |
| Uniform(p)     | $θ = X_n$                |      

## Ước lượng nhiều tham số 

Lần này, thay vì chỉ ước lượng một thì chúng ta ước lượng nhiều tham số một chút. Các bước làm sẽ tương tự như ở trên, chỉ khác ở chỗ: _ta sẽ phải đạo hàm riêng nhiều lần_

__Ví dụ__

Cho  $x_1,x_2,...,x_n$∼  N(μ, $σ^2$). Dùng phương pháp Ước lượng khả năng cực đại(MLE) để ước lượng μ và $σ^2$.   

_Giải_

- phân phối xác suất: $\frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{(x-μ)^2}{2\sigma^2}}$

- likelihood:

$$L(μ ,σ^2) = ∏_{i=1}^{n}\frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{(x-μ)^2}{2\sigma^2}}$$

- log likelihood:

Đặt $l = log(L(μ ,σ^2))$, ta có:

$$l = ∑_{i=1}^{n} log[\frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{(x-μ)^2}{2\sigma^2}}]$$

$$= ∑_{i=1}^{n} log(\frac{1}{\sigma \sqrt{2\pi}}) + ∑_{i = 1}^{n}log[ e^{-\frac{(x-μ)^2}{2\sigma^2}}]$$

$$= - \frac{n}{2} log(\sigma^2) - - \frac{n}{2} log(2\pi) - \frac{1}{2\sigma^2} ∑_{i=1}^{n} (x_i - μ )^2$$

- Đạo hàm $l$ theo 2 biến μ  và $σ^2$

    $$\frac{\partial l}{\partial μ} = \frac{1}{\sigma^2} ∑_{i=1}^{n} (x_i - μ)$$

    $$\frac{\partial l}{\partial σ^2} = -\frac{n}{2\sigma^2} + \frac{1}{2(\sigma^2)^2}∑_{i=1}^{n} (x_i - μ )^2$$


- likelihood đạt cực đại


$$
⟺ \begin{cases}
    \frac{\partial l}{\partial μ} = 0 \\
    \frac{\partial l}{\partial σ^2} = 0
\end{cases}
$$

$$
⟺ \begin{cases}
    μ = \bar{x} \\
    σ^2 = \frac{1}{n}∑_{i=1}^{n} x_i^2 - \bar{x}^2
\end{cases}
$$

## ứng dụng

MLE có rất nhiều ứng dụng (chủ yếu là dành cho dân thống kê và dân nghiên cứu) chẳng hạn như lý giải một hiện thượng nào đó có liên quan tới xác xuất.

Ví dụ: 

- Tung đồng xu 10 lần. Khi đó, ta coi đồng xu là một biến ngẫu nhiên X chỉ nhận một trong 2 giá trị là "sấp" hoặc "ngửa".
Việc tung 10 lần mang hàm ý là chúng ta lấy ra 10 mẫu dữ liệu. Và thoạt nhìn thì ta cũng biết phân phối dữ liệu là phân phối nhị thức (binominal). 
Bộ tham số của phân phối lúc này là $θ(n, p)$, trong đó n là số phép thử và p là xác xuất cho mỗi lần thử.

MLE: $L(n,p) = f(x | n, p) = P( X =  x_1,x_2,...x_n) = \binom{n}{x} p^x (1−p)^{n−x}$


Thấy quen thuộc chứ ? Đây chính công thức xác suất của phân phối nhị thức

Hoặc, bạn có 10 con xúc xắc không đồng chất (có con 4 mặt, 5 mặt, 6 mặt,...) và bạn muốn biết con nào cho ra xác suất 7 mạth
