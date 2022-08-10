from py2neo import *
import numpy as np
import pandas as pd

if __name__ == "__main__":
    graph = Graph('http://localhost:7474', auth=('neo4j', 'wyj12369'))
    graph.delete_all()
    df = pd.read_csv("C:/Users/wyj/Desktop/test.csv")

    df = df.dropna(axis=0, how="all").fillna(axis=0, method="ffill")
    df["keywords"] = np.array([x.split("+") for x in list(np.array(df["keywords"], dtype='object'))], dtype="object")
    mat = np.array(df)[1:, :]

    key_word_node_dic = {}
    key_word_list = []
    for temp_key_word_list in mat[:, -1]:
        key_word_list += temp_key_word_list
    key_word_list = list(set(list(key_word_list)))
    for key_word in key_word_list:
        key_word_node_dic[key_word] = Node("keyword", name=key_word)

    paper_node_list = [Node("paper", name=paper[1], ) for paper in mat]

    df = pd.read_csv("C:/Users/wyj/Desktop/国内外MAS+DP团队汇总.csv")
    temp_mat = np.array(df)[1:, :]
    team_node_list = [Node("team", branch=branch_team_expert[0], team=branch_team_expert[1],
                           expert=branch_team_expert[2]) for branch_team_expert in temp_mat]

    main_node = Node("root", name="MAS+DP")
    graph.create(main_node)

    team_n = -1
    team_list = []
    main_paper_list = []
    main_paper_node_list = []
    main_paper_node = None
    for paper in mat:
        if paper[0] not in team_list:
            team_n += 1
            graph.create(team_node_list[team_n])
            graph.create(Relationship(main_node, "团队", team_node_list[team_n]))
            print(team_node_list[team_n])
            team_list.append(paper[0])
        if paper[1] == 0 or '0':
            main_paper_node = Node("paper", name=paper[1], article=paper[3],
                                   author=paper[4], year=paper[5], keyword="+".join(paper[6]))
            graph.create(main_paper_node)
            graph.create(Relationship(team_node_list[team_n], "发表", main_paper_node))
            main_paper_list.append(paper[1])
            main_paper_node_list.append(main_paper_node)
            for one_key_word in paper[6]:
                graph.create(Relationship(main_paper_node, "关键词", key_word_node_dic[one_key_word]))
                graph.create(Relationship(key_word_node_dic[one_key_word], "论文", main_paper_node))
            continue
        paper_node = Node("paper", name=paper[2], article=paper[3], author=paper[4],
                          year=paper[5],  keyword="+".join(paper[6]))\
            if paper[2] not in main_paper_list else main_paper_node_list[main_paper_list.index(paper[2])]
        graph.create(paper_node)
        graph.create(Relationship(main_paper_node, "引用", paper_node))
        for one_key_word in paper[6]:
            graph.create(Relationship(paper_node, "关键词", key_word_node_dic[one_key_word]))
            graph.create(Relationship(key_word_node_dic[one_key_word], "论文", paper_node))
