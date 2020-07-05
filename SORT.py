############################
#
#       实现自己的排序算法
#      （默认从小到大排序）
#       截止日期：20-7-5
############################


#           时间复杂度    空间复杂度    最好情况    最坏情况    内外排序    稳定性

# 冒泡排序      n^2          1            n         n^2         内         稳
# 选择排序      n^2          1            n^2       n^2         内        不稳
# 插入排序      n^2          1            n         n^2         内         稳  #对于近乎有序的数据效率极高

# 希尔排序     nlog2n        1         n(log2n)    n(log2n)     内        不稳
# 归并排序     nlogn         n         n(logn)     n(logn)      外         稳
# 快速排序     nlogn        logn       n(logn)      n^2         内        不稳
# 堆排序       nlogn         1         n(logn)     n(logn)      外        不稳    未完成

# 计数排序      n+k          k           n+k         n+k        外         稳
# 桶排序        n+k         n+k          n+k         n^2        外         稳    未完成
# 基数排序      n*k         n+k          n*k         n*k        外         稳    未完成

from random import randint
import time
import re

# 冒泡排序 两两比较 大的沉底（最后）
# 若某次循环未发生交换则退出
def sort_bubble(nums):
	n = len(nums)
	if n<1:
		raise ValueError('Error!  is empty')
	if n == 1:
		return nums
	for i in range(n):
		t = 0       # 标记此次是否发生交换   某次循环未交换则退出
		for j in range(1,n):
			if nums[j-1]>nums[j]:
				nums[j-1],nums[j]=nums[j],nums[j-1]
				t = 1
		if t==0:
			break
	return nums

## 标记最后发生交换的位置，后面的排序过程中不考虑这之后的数据
def sort_bubble_2(nums):
	n = len(nums)
	if n<1:
		raise ValueError('Error!  is empty')
	if n == 1:
		return nums
	new_n = 9999    # 只是为了进入while 随便给个数字
	while(new_n > 0):
		new_n = 0
		for j in range(1,n):
			if nums[j-1]>nums[j]:
				nums[j-1],nums[j]=nums[j],nums[j-1]
				new_n = j
		n = new_n   # 标记最后发生交换的位置，后面的排序过程中不考虑这之后的数据
	return nums

# 选择排序 每一次选择最小的放前面
def sort_select(nums):
	n = len(nums)
	if n<1:
		raise ValueError('Error!  is empty')
	if n == 1:
		return nums
	for i in range(n):
		t = i   # 当前最小值的下标
		for j in range(i,n):
			if nums[j]<nums[t]:
				t = j
		nums[i],nums[t] = nums[t],nums[i]
	return nums

# 插入排序 每次循环增加一个数字到有序数组中
def sort_insert(nums):
	n = len(nums)
	if n < 1:
		raise ValueError('Error!  is empty')
	if n == 1:
		return nums
	for i in range(1,n):
		for j in range(i):
			if nums[j] > nums[i]:
				nums[i],nums[j]=nums[j],nums[i]
	return nums

## 修改j从后往前判断
def sort_insert_2(nums):
	n = len(nums)
	if n < 1:
		raise ValueError('Error!  is empty')
	if n == 1:
		return nums
	for i in range(1,n):
		for j in range(i,0,-1):
			if nums[j] < nums[j-1]:
				nums[j-1], nums[j] = nums[j], nums[j-1]
	return nums

## 增加哨兵机制 减少操作次数
def sort_insert_3(nums):
	n = len(nums)
	if n < 1:
		raise ValueError('Error!  is empty')
	if n == 1:
		return nums
	for i in range(1,n):
		t = nums[i]
		is_break = 0
		for j in range(i, 0, -1):           # 左闭右开
			if nums[j-1] > t:
				nums[j] = nums[j-1]
			else:
				is_break = 1
				break
		# nums[j] = t
		# python的循环结束后不会再-1,因此这里要区分是循环结束还是中途退出
		if is_break:
			nums[j] = t
		else:
			nums[0] = t
	return nums

# 希尔排序
# 按照间隔k排序 按k/2排序 按k/4排序……
def sort_shell(nums):
	n = len(nums)
	if n < 1:
		raise ValueError('Error!  is empty')
	if n == 1:
		return nums
	k = n//2
	while(k > 0):
		# 基本类似于分组直插
		for i in range(k):
			for j in range(i, n-k, k):
				if nums[j+k] < nums[j]:
					nums[j + k], nums[j] = nums[j], nums[j + k]
		k = k // 2
	# 最后基本有序，调用直插
	nums = sort_insert_3(nums)
	return nums

# 归并排序 分治 自顶向下
def sort_merge(nums):
	n = len(nums)
	if n < 1:
		raise ValueError('Error!  is empty')
	if n <= 15:     # 数据较少用直插优化
		return sort_insert_3(nums)
	mid = n//2
	# 对左右归并排序
	left  = sort_merge(nums[:mid])      # 下标[0 ~ mid-1]
	right = sort_merge(nums[mid:])      #    [mid ~ n-1]
	return MERGE(left, right)

# 自底向上的归并排序
# 通过修改MERGE，即向MERGE中传入原数据和左右边界
# 可以达到对链表进行nlogn级别的排序（自底向上归并的亮点）
# 自底向上比自顶向下效率略差一点
def sort_merge_2(nums):
	n = len(nums)
	if n < 1:
		raise ValueError('Error!  is empty')
	size = 1
	while(size < n):
		i = 0
		while(i + size < n):
			MIN_r = min(i+size + size-1, n-1) + 1  #区间是左闭右开，所以最后要+1
			nums[i : MIN_r] = MERGE(nums[i : i+size],
			                        nums[i+size : MIN_r] )
			i += 2*size
		size += size
	return nums

# 归并排序子步骤，将两个有序的数组合并
def MERGE(left, right):
	temp = []
	i, j = 0, 0
	while (i < len(left) and j < len(right)):  # 一一比较 小的先进
		if left[i] < right[j]:
			temp.append(left[i])
			i += 1
		else:
			temp.append(right[j])
			j += 1
	temp += left[i:]  # 有一个到尾就把后面的直接加入
	temp += right[j:]
	return temp

# 快排
# 从后找小往前放，从前找大往后放
# 每次选择第一个数字做标准 每次操作直接覆盖
def sort_quick(nums):
	n = len(nums)
	if n < 1:
		raise ValueError('Error!  is empty')

	# 每次选择第一个数字做标准 每次操作直接覆盖
	def __quick(l, r):
		if l>=r:
			return
		if n <= 15:  # 数据较少用直插优化
			return sort_insert_3(nums[l:r+1])
		# 从两边开始往中间找，两个指针相遇即结束
		i, j = l, r
		t = nums[i]
		while i<j:
			while i<j and nums[j]>=t:       # 重点！！！相等时可以放任意一边 但一定要加
				j -= 1
			nums[i] = nums[j]
			while i<j and nums[i]<t:
				i += 1
			nums[j] = nums[i]
		nums[i] = t
		__quick(l, i-1)
		__quick(i+1, r)
		return nums

	nums = __quick(0, n-1)
	return nums

# 固定取第一个数字做标准，每次操作为交换
# 存在问题：对于近乎有序的数组将退化成n2级别
def sort_quick_bobo(nums):
	n = len(nums)
	if n < 1:
		raise ValueError('Error!  is empty')

	# 第一个数字做标准，每次操作为交换
	def __quick(l, r):
		if l>=r :
			return
		if n <= 15:  # 数据较少用直插优化
			return sort_insert_3(nums[l:r+1])
		i, j = l, l+1   ##重点1 两个初值不同 一个是l 一个是l+1
		t = nums[l]
		while j <= r:
			if nums[j] < t:
				nums[j], nums[i+1] = nums[i+1], nums[j]  ##重点2 与i+1交换不是i
				i += 1  # 当前小于标准的右边界
			j += 1  # 当前判断的位置
		nums[l], nums[i] = nums[i], nums[l]
		__quick(l, i-1)
		__quick(i+1, r)
		return nums

	nums = __quick(0, n - 1)
	return nums

# 随机取标准，每次操作为交换
# 解决问题：随机取标准以应对近乎有序的数组不至于退化成n2级别（此时变成一种随机算法）
# 存在问题：数组内存在大量重复元素时效果极差
def sort_quick_bobo_2(nums):
	n = len(nums)
	if n < 1:
		raise ValueError('Error!  is empty')

	# 随机取标准，每次操作为交换
	def __quick(l, r):
		if l>=r :
			return
		if n <= 15:  # 数据较少用直插优化
			return sort_insert_3(nums[l:r+1])
		i, j = l, l+1   ##重点1 两个初值不同 一个是l 一个是l+1
		mid = randint(l, r)
		nums[l], nums[mid] = nums[mid], nums[l]
		t = nums[l]
		while j <= r:
			if nums[j] < t:
				nums[j], nums[i+1] = nums[i+1], nums[j]  ##重点2 与i+1交换不是i
				i += 1  # 当前小于标准的右边界
			j += 1  # 当前判断的位置
		nums[l], nums[i] = nums[i], nums[l]
		__quick(l, i-1)
		__quick(i+1, r)
		return nums

	nums = __quick(0, n - 1)
	return nums

# 与之前的sort_quick类似从前找大，从后找小。但又有所不同，需要注意
# (称为双路快排，数组被分为<= >= 两部分 )
# 解决问题：数组内存在大量重复元素效率仍较好
def sort_quick_bobo_3(nums):
	n = len(nums)
	if n < 1:
		raise ValueError('Error!  is empty')

	# 随机取标准，每次操作为交换
	def __quick(l, r):
		if l>=r :
			return
		if n <= 15:  # 数据较少用直插优化
			return sort_insert_3(nums[l:r+1])
		i, j = l+1, r
		mid = randint(l, r)
		nums[l], nums[mid] = nums[mid], nums[l]
		t = nums[l]
		while 1:
			# 往后找大
			while(i <= j and nums[i] < t):  ### 重点！！这里必须是<=和>=
				i += 1
			# 往前找小
			while(j >= i and nums[j] > t):
				j -= 1
			# 此时与标准t相同的元素被分摊到两边，不至于堆在一起
			if i >= j:
				break
			nums[i], nums[j] = nums[j], nums[i]
			i += 1
			j -= 1
		nums[l], nums[j] = nums[j], nums[l]
		__quick(l, j-1)
		__quick(j+1, r)
		return nums

	nums = __quick(0, n - 1)
	return nums

# 进一步优化存在大量重复元素时的效率
# (三路快排 数组被分为< = > 三部分)
def sort_quick_bobo_4(nums):
	n = len(nums)
	if n < 1:
		raise ValueError('Error!  is empty')

	# 随机取标准，每次操作为交换
	def __quick(l, r):
		if l>=r :
			return
		if n <= 15:  # 数据较少用直插优化
			return sort_insert_3(nums[l:r+1])
		i, k, j= l, l+1, r
		## [l, i-1] < t      i是左边第一个不大于t的位置
		## [i, j] == t
		## [j+1, r] > t      j是右边第一个不小于t的位置
		mid = randint(l, r)
		nums[l], nums[mid] = nums[mid], nums[l]
		t = nums[l]
		while 1:
			if nums[k] == t:
				k += 1
			elif nums[k] > t:
				nums[j], nums[k] = nums[k], nums[j]
				j -= 1
			else: #nums[k] < t
				nums[i], nums[k] = nums[k], nums[i]
				i += 1
				k += 1
			if k > j:   ### 重点 这里不能加=号！！！！
				break
		__quick(l, i-1)
		__quick(j+1, r)
		return nums

	nums = __quick(0, n - 1)
	return nums

# 堆排 利用堆这种数据结构
def sort_heap(nums):

	return nums

# 计数排序  典型空间换时间
# 统计从最小值到最大值之间每个数字出现了多少次，然后按从小到大的顺序依次输出
def sort_conut(nums):
	n = len(nums)
	if n < 1:
		raise ValueError('Error!  is empty')
	if n == 1:
		return nums
	MIN = min(nums)
	MAX = max(nums)
	num = [0] * (MAX-MIN+1)         # 原数组数据范围     ######## 记得+1
	for i in range(n):              # 统计范围内每个数出现次数 即计数
		num[nums[i]-MIN] += 1
	nums = []
	for i in range(MAX-MIN+1):                          ####### 记得循环范围要+1 把最大的数也要加进去
		if num[i]:
			while num[i]:
				nums.append(i+MIN)
				num[i] -= 1
	return nums

# 桶排序
def sort_bucket(nums):

	return nums

# 基数排序
def sort_radix(nums):

	return nums

# 随机生成n个[rangeL, rangeR]之间的数字
def geneArr(n, rangeL, rangeR):
	if rangeR < rangeL:
		raise ValueError("range is error!")
	arr = [None] * n
	for i in range(n):
		arr[i] = randint(rangeL, rangeR)
	return arr

def test(arr, sort_title, is_print):
	num = arr.copy()  # 如果不复制 则经过一个排序算法之后原数组已经原地有序
	_sort_title = eval('sort_title')  # <function sort_select at 0x00000196EEE2A0D8>

	start = time.time()
	num = _sort_title(num)
	if is_print:
		print(num)
	t = time.time() - start

	is_order = True
	for i in range(1, len(arr)):
		if num[i-1] > num[i]:
			is_order = False
			break

	# title = re.match(r"(?<='function ').*?(?=' at')", _sort_title)
	title = re.findall(r'sort_(.*?) at', str(_sort_title))
	# 左对齐，占位18，字符串
	STR = '%-18s' % str(title) + '\t' + str(is_order)
	print(STR + ': '  + str(t) + 's')


if __name__ == '__main__':

	n = 10000
	L,R = [1, n]
	is_print = 0
	num_ = geneArr(n, L, R)
	# num_ = [4, 4, 8, 5, 3, 5, 1, 6, 8, 10]
	if is_print:
		print(num_)

	test(num_, sort_bubble, is_print)
	test(num_, sort_bubble_2, is_print)     #冒泡2较快
	test(num_, sort_select, is_print)
	test(num_, sort_insert, is_print)
	test(num_, sort_insert_2, is_print)
	test(num_, sort_insert_3, is_print)     #直插3较快

	test(num_, sort_merge, is_print)        # 归并1较快
	test(num_, sort_merge_2, is_print)      # 两种不同的方式（这里并不是优化）
	test(num_, sort_quick, is_print)
	test(num_, sort_quick_bobo, is_print)
	test(num_, sort_quick_bobo_2, is_print)
	test(num_, sort_quick_bobo_3, is_print)
	test(num_, sort_quick_bobo_4, is_print)
	test(num_, sort_shell, is_print)
	test(num_, sort_conut, is_print)
