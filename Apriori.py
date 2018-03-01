def load_data_set():
    """
    Load a sample data set (From Data Mining: Concepts and Techniques, 3th Edition)
    Returns: 
        A data set: A list of transactions. Each transaction contains several items.
    """
    # 这个函数其实就是数据挖掘 概念与技术 书上的例子
    data_set = [['l1', 'l2', 'l5'], ['l2', 'l4'], ['l2', 'l3'],
            ['l1', 'l2', 'l4'], ['l1', 'l3'], ['l2', 'l3'],
            ['l1', 'l3'], ['l1', 'l2', 'l3', 'l5'], ['l1', 'l2', 'l3']]
    return data_set
def create_C1(data_set):
    """
    Create frequent candidate 1-itemset C1 by scaning data set.
    Args:
        data_set: A list of transactions. Each transaction contains several items.
    Returns:
        C1: A set which contains all frequent candidate 1-itemsets
    """
    # 产生1项集
    C1 = set()
    for t in data_set:
        for item in t:
            # use frozenset to ensure 'item' can be used as key of dictionary
            # C1.add(frozenset([item]))
            # frozenset()的输入必须是一个列表
            item_set = frozenset([item])
            C1.add(item_set)
    return C1
def is_apriori(Ck_item, Lksub1):
    """
    Judge whether a frequent candidate k-itemset satisfy Apriori property.
    Args:
        Ck_item: a frequent candidate k-itemset in Ck which contains all frequent
                 candidate k-itemsets.
        Lksub1: Lk-1, a set which contains all frequent candidate (k-1)-itemsets.
    Returns:
        True: satisfying Apriori property.
        False: Not satisfying Apriori property.
    """
    # 先验性质的判断：频繁子集的非空子集一定的是频繁的
    # 这一步相当于是剪枝操作
    for item in Ck_item:
        # python中set和frozenset方法和区别
        # https://www.cnblogs.com/panwenbin-logs/p/5519617.html
        # 集合可以之间加减运算，这一步相当与一个个去掉元素，然后判断子集是否是在k-1项集中，如果不满足先验性质，则说明此k项集元素应该去掉
        sub_Ck = Ck_item - frozenset([item])
        if sub_Ck not in Lksub1:
            return False
    return True
def create_Ck(Lksub1, k):
    """
    Create Ck, a set which contains all all frequent candidate k-itemsets
    by Lk-1's own connection operation.
    Args:
        Lksub1: Lk-1, a set which contains all frequent candidate (k-1)-itemsets.
        k: the item number of a frequent itemset.
    Return:
        Ck: a set which contains all all frequent candidate k-itemsets.
    """
    # 生成Ck候选集的方式，判断前面k-1项是否相同，然后组合最后一项
    Ck = set()
    len_Lksub1 = len(Lksub1)
    # 这一步将set转换为list是为了遍历方便么？
    list_Lksub1 = list(Lksub1)
    # 两个嵌套的for循环完成ck-1项的遍历
    # apriori算法的缺点也就是在这里
    for i in range(len_Lksub1):
        for j in range(i+1, len_Lksub1):
            l1 = list(list_Lksub1[i])
            l2 = list(list_Lksub1[j])
            # 防止set集合list化之后不能进行排序
            l1.sort()
            l2.sort()
            if l1[0:k-2] == l2[0:k-2]:
                # 集合set才有|这个求并集的方法 
                Ck_item = list_Lksub1[i] | list_Lksub1[j]
                # pruning
                if is_apriori(Ck_item, Lksub1):
                    Ck.add(Ck_item)
    return Ck
def generate_Lk_by_Ck(data_set, Ck, min_support, support_data):
    """
    Generate Lk by executing a delete policy from Ck.
    Args:
        data_set: A list of transactions. Each transaction contains several items.
        Ck: A set which contains all all frequent candidate k-itemsets.
        min_support: The minimum support.
        support_data: A dictionary. The key is frequent itemset and the value is support.
        这个是一个全局量，从始至终都是存在的状态
    Returns:
        Lk: A set which contains all all frequent k-itemsets.
    """
    # 在候选集合Ck中过滤得到满足最小支持度的Lk
    # 在item_count的基础上，过滤得到的最终的模式与其频数
    Lk = set()
    # item_count 用于储存所有事务中包含模式和对应的频数的字典数据结构
    item_count = {}
    for t in data_set:
        for item in Ck:
            # 利用集合的issubset的函数判断item是不是事务t的子集
            # 如果判断为真，则在item模式上的count上+1
            if item.issubset(t):
                if item not in item_count:
                    item_count[item] = 1
                else:
                    item_count[item] += 1
    # 支持度计算公式的基数
    t_num = float(len(data_set))
    for item in item_count:
        if (item_count[item] / t_num) >= min_support:
            Lk.add(item)
            support_data[item] = item_count[item] / t_num
    return Lk
def generate_L(data_set, k, min_support):
    """
    Generate all frequent itemsets.
    Args:
        data_set: A list of transactions. Each transaction contains several items.
        k: Maximum number of items for all frequent itemsets.
        min_support: The minimum support.
    Returns:
        L: The list of Lk.
        support_data: A dictionary. The key is frequent itemset and the value is support.
    """
    support_data = {}
    # 得到1项集
    C1 = create_C1(data_set)
    # 得到了一项集对应的元素的频数，以及增加了support_data中的元素数量
    L1 = generate_Lk_by_Ck(data_set, C1, min_support, support_data)
    # 集合set的复制，Lksub1
    Lksub1 = L1.copy()
    # L是存储所有Lk的模式与其频数
    L = []
    L.append(Lksub1)
    # k是函数给出的，说明模式寻找限制的项集数
    for i in range(2, k+1):
        # 生成待选k项集
        Ci = create_Ck(Lksub1, i)
        Li = generate_Lk_by_Ck(data_set, Ci, min_support, support_data)
        # 下一个步骤，替换了Lksub1的值，确保可以递归进行下去求Ck
        Lksub1 = Li.copy()
        L.append(Lksub1)
    return L, support_data
def generate_big_rules(L, support_data, min_conf):
    """
    Generate big rules from frequent itemsets.
    Args:
        L: The list of Lk.
        support_data: A dictionary. The key is frequent itemset and the value is support.
        min_conf: Minimal confidence.
    Returns:
        big_rule_list: A list which contains all big rules. Each big rule is represented
                       as a 3-tuple.
    """
    # 这个函数应该是为了满足最小置信度而做出来的函数
    big_rule_list = []
    sub_set_list = []
    for i in range(0, len(L)):
        for freq_set in L[i]:
            # 刚开始就是空集，很奇怪
            # 这个方法相当于每次添加一个lk中的元素，然后遍历列表，接着判断模式之间是否有包含集合关系，根据置信度的计算公式可以发现“先验”的含义
            for sub_set in sub_set_list:
                if sub_set.issubset(freq_set):
                    conf = support_data[freq_set] / support_data[freq_set - sub_set]
                    # 很有意思！
                    big_rule = (freq_set - sub_set, sub_set, conf)
                    if conf >= min_conf and big_rule not in big_rule_list:
                        # print freq_set-sub_set, " => ", sub_set, "conf: ", conf
                        big_rule_list.append(big_rule)
            sub_set_list.append(freq_set)
    return big_rule_list
if __name__ == "__main__":
    """
    Test
    """
    # 加载数据集
    data_set = load_data_set()
    # 输出满足支持度的频繁模式列表L与支持度support_data
    L, support_data = generate_L(data_set, k=3, min_support=0.2)
    # 输出满足置信度的频繁模式
    big_rules_list = generate_big_rules(L, support_data, min_conf=0.7)
    for Lk in L:
        print "="*50
        print "frequent " + str(len(list(Lk)[0])) + "-itemsets\t\tsupport"
        print "="*50
        for freq_set in Lk:
            print freq_set, support_data[freq_set]
    print
    print "Big Rules"
    for item in big_rules_list:
        print item[0], "=>", item[1], "conf: ", item[2]