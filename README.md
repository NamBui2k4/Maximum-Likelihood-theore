Bài viết này được kham khảo từ nguồn: 

- [_phamdinhkhanh.github_](https://phamdinhkhanh.github.io/deepai-book/ch_ml/NaiveBayes.html)

- [_kierandg.blogspot.com_](https://kierandg.blogspot.com/2016/05/so-luoc-log-likelihood-maximum-likelihood-estimation.html)

- [_dangnguyenit.blogspot.com_](https://dangnguyenit.blogspot.com/2018/10/uoc-luong-hop-ly-cuc-aimaximum.html)

# Ước lượng hợp lý tối đa (Maximum Likelihood Estimation - MLE)

Trong thống kê, dữ liệu thường được diễn tả thông qua những __phân phối xác suất__. Phân phối xác suất thường được đặc trưng bởi những tham số nhất định. 
Ví dụ, đối với phân phối chuẩn, tham số đặc trưng chính là cặp trung bình ($μ $) và phương sai ($σ^2$). Đối với phân phối Poisson thì tham số đặc trưng là λ

Tuy nhiên, có những tình huống khó khăn khi chúng ta download một vài bộ dữ liệu từ nguồn nào đó mà không biết rõ phân phối của nó là gì. Và nếu không xác
định được phân phối thì việc làm thống kê dữ liệu sẽ rất khó khăn. Đó là lúc mà giải pháp **ước lượng hợp lý tối đa** (Maximum Likelihood Estimation - viết tắt là MLE) ra đời. 

Tuy nhiên, trước hết, ta cần hiểu rõ __hàm hợp lý(Likelihood function)__ là gì.

## Định nghĩa hàm hợp lý

    Trong thống kê, Likelihood function - gọi là
    hàm hợp lý, được hiểu là một hàm đo lường
    mức độ "phù hợp" của một bộ tham số cụ thể với
    dữ liệu quan sát được. 
\
Giả sử biến ngẫu nhiên  X  tuân theo một phân phối nào đó được mô tả bởi bộ tham số $θ ( θ_1,θ_2,...,θ_k )$ mà ta chưa biết.

Hàm hợp lý của phân phối có dạng:

- $L(θ)= f(x_1,x_2,...,x_n | θ_1,θ_2,...,θ_k)$ (nếu  X  là biến liên tục)
- $L(θ)= p(x_1,x_2,...,x_n | θ_1,θ_2,...,θ_k)$ (nếu  X  là biến rời rạc)

Trong đó:
- $x_1,x_2,...,x_n$ là mẫu dữ liệu thu được từ tổng thể
- $θ_1,θ_2,...,θ_k$ là tập hợp tham số của phân phối dữ liệu

Hàm hợp lý có thể được hiểu là xác suất để các sự kiện  $x_1,x_2,...,x_n$ cùng xảy ra (xác suất đồng thời), với điều kiện $( θ_1,θ_2,...,θ_k )$. 
Như vậy, cách gọi "hàm hợp lý" ở đây chính là xác suất có điều kiện. 

## Định nghĩa ước lượng hợp lý tối đa (Maximum Likelihood definition)

    Trong thống kê, ước lượng hợp lý tối đa là một
    phương pháp ước lượng tham số của phân phối dữ
    liệu bằng cách tối đa hoá hàm hợp lý sao cho
    dưới giả định của thống kê thì dữ liệu
    trở nên phù hợp nhất.
\
Hay nói cách khác, ước lượng hợp lý tối đa là đi tìm bộ tham số θ  để xác suất đồng thời xảy ra các kết quả $x_1,x_2,...,x_n$ là cao nhất.

Hoặc nói đơn giản hơn, ước lượng hợp lý tối đa là tìm θ để likelihood đạt max

Tuy nhiên, điều này không hề đơn giản. Vì các sự kiện $x_1,x_2,...,x_n $xảy ra đồng thời nên ta phải giải quyết bài toán "tính xác suất để X nhận được tất cả các giá trị $x_1,x_2,...,x_n $". Mặt khác, ta đã nói rằng  hàm likelihood được viết gần giống hàm xác xuất. Điều đó có nghĩa là, ta có thể biến đổi likelihood giống như  biến đổi một hàm xác suất. Nếu X liên tục, ta biến đổi như sau:

 $$L(θ) = f(x_1,x_2,...,x_n | θ) = f(x_1|θ).f(x_2|θ)...f(x_n|θ)$$

Phép biến đổi trên chính là ứng dụng của __hàm phân phối tích lũy của các biến liên tục__ $x_1,x_2,...,x_n$(xem lại trong file CDF.md )


Nếu X rời rạc, ta biến đổi:

$$L(θ)= p(x_1,x_2,...,x_n | θ_1,θ_2,...,θ_k) = p(x_1|θ) + p(x_2|θ) +...+ p(x_n|θ_k)$$

Tương tự, đây cũng là ứng dụng của __hàm phân phối tích lũy của các biến rời rạc__ $x_1,x_2,...,x_n$(xem lại trong file CDF.md ). Tuy nhiên, trong thực tế người ta ít khi sử dụng loại likelihood này

Rõ ràng, việc tính toán likelihood gặp rất nhiều trở ngại. Đấy là chưa kể ta không biết θ sẽ chứa cái gì (tùy thuộc vào loại phân phối mà θ có thể chứa μ, σ, λ) để mà ước lượng

## Ước lượng một tham số

Giả sử bài toán yêu cầu ước lượng hợp lý cho một tham số trong bộ θ (chẳng hạn như phân phối B(p) chỉ có một tham số là p, hay N(μ,σ) có 2 tham số nhưng ta đã biết trước 1 trong 2 và yêu cầu ước lượng cho cái còn lại). Thực hiện cái bước:

1. Xác định likelihood
2. Dùng logarit đưa likelihood về dạng tổng: đặt $u(θ) = log(L(θ))$.  
3. Đạo hàm một biến lên $u$, ta được : $u(θ)' = log(L(θ))'$
4. likelihood đạt cực đại
 ⇔ $u(θ)$đạt cực đại
 ⟺ $u(θ)' = 0$
 ⟺ tham số trong $θ$bằng một giá trị nào đó
5. kết luận $θ$

__Ví dụ__

Cho  $x_1,x_2,...,x_n$∼  Bernoulli(p). Dùng phương pháp Ước lượng hợp lý cực đại(MLE) để ước lượng tham số p .

_Giải:_

phân phối Bernoulli(p) là phân phối xác suất của biến liên tục x với hàm khối xác suất là: $f(x|θ) = p^{x}(1−p)^{1−x}  $

_Lưu ý: Cách viết $f(x|θ)$nhằm mục đích liên hệ với chủ đề likelihood_

1. Xác định Likelihood: 
$L(p) = f(x_1,x_2,...,x_n | p) = f(x_1|p).f(x_2|p)...f(x_n|p)$

$$=  ∏_{i = 1}^{n} p^{x_i}(1−p)^{1−x_i}  $$

2. Dùng logarit đưa likelihood về dạng tổng

Đặt $u(p) = log(L(p)) $, ta có:

$$u(p) = log(∏_{i = 1}^{n} p^{x_i}(1−p)^{1−x_i} ) $$
$$= ∑_{i = 1}^{n} log(p^{x_i}(1−p)^{1−x_i}) $$
$$= ∑_{i=1}^{n} log(p^{x_i})+ ∑_{i=1}^{n}log(1−p)^{1−x_i}$$
$$= ∑_{i=1}^{n} x_i×log(p) + ∑_{i=1}^{n}(1−x_i)×log(1−p) $$

Đặt  $∑_{i=1}^{n} x_i = t$, ta coi t như một hằng số thì phương trình trở thành:

$u(p) = t × log(p) + (n - t) × log(1-p)$

3. đạo hàm: $u'(p) = t×\frac{1}{p} +(n−t)× \frac{−1}{1−p}$

4. likelihood đạt cực đại
⟺ $u(p)$đạt cực đại
⟺ $u(p)' = 0 $
⇔ $t×\frac{1}{p} +(n−t)× \frac{−1}{1−p}=0$
⇔ $p = \frac{1}{n} ∑_{i=1}^{n}x_i$

5. Kết luận:  $p = \frac{1}{n} ∑_{i=1}^{n}x_i$là ước lượng hợp lý tối đa cần tìm

Nếu ta để ý thì  $p = \frac{1}{n} ∑_{i=1}^{n}x_i$chính là trung bình mẫu $\bar{x}$của mẫu chứa các giá trị $x_1, x_2,...,x_n,$

__Bảng tóm tắt công thức ước lượng hợp lý tối đa của một số phân phối__

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

Bài toán này thường xảy khi ta phải đi tìm một tổng thể ban đầu mà ta chưa biết, nhưng ta đã có được mẫu dữ liệu của nó.

Nó giống như việc đi tìm chiều cao trung bình của dân số nhưng lại không thể lấy hàng triệu người trong dân số đó để đo đạc. Bù lại, ta có thể lấy ngẫu nhiên 100 người trong dân số đó và thống kê ước lượng, lặp đi lặp lại rất nhiều lần cho đến khi có được kết quả hợp lý nhất.

Giả sử yêu cầu ước lượng hợp lý cực đại cho tất cả tham số $θ_1,θ_2,...,θ_k$trong bộ θ. Các bước làm sẽ tương tự như ước lượng một tham số ở trên, chỉ khác ở chỗ: _ta sẽ phải đạo hàm k lần, mỗi lần ta coi một tham số là một biến và các tham số còn lại là hằng số_

__Ví dụ__

Cho  $x_1,x_2,...,x_n$∼  N(μ, $σ^2$). Dùng phương pháp Ước lượng hợp lý cực đại(MLE) để ước lượng μ và $σ^2$.   

_Giải_

- Hàm khối xác suất: $\frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{(x-μ)^2}{2\sigma^2}}$

- likelihood:

$$L(μ ,σ^2) = ∏_{i=1}^{n}\frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{(x-μ)^2}{2\sigma^2}}$$

- log likelihood:

Đặt $l = log(L(μ ,σ^2))$, ta có:

$$l = ∑_{i=1}^{n} log[\frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{(x-μ)^2}{2\sigma^2}}]$$

$$= ∑_{i=1}^{n} log(\frac{1}{\sigma \sqrt{2\pi}}) + ∑_{i = 1}^{n}log[ e^{-\frac{(x-μ)^2}{2\sigma^2}}]$$

$$= - \frac{n}{2} log(\sigma^2) - - \frac{n}{2} log(2\pi) - \frac{1}{2\sigma^2} ∑_{i=1}^{n} (x_i - μ )^2$$

- Đạo hàm $l$theo 2 biến μ  và $σ^2$

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
Bộ tham số của phân phối lúc này là $θ(n, p)$, trong đó n là số phép thử và p là xác xuất của một kết quả mỗi lần thử.

MLE: $L(n,p) = f(x | n, p) = \binom{n}{x} p^x (1−p)^{n−x}$

hoặc có thể viết tắt là

$$L(n, p)=P( X =  x_1,x_2,...x_n) $$

Thấy quen thuộc chứ ? Đây chính công thức xác suất của phân phối nhị thức
