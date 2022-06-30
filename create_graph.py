import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import distance

class GraphFromImage:
    def __init__(self, image_path):
        self.image_path = image_path

    def make_graph(self):
        G = nx.Graph()
        points = self.points_from_image()
        for i, point in enumerate(points):
            G.add_node(i, pos=point)

        N_NEIGHBOURS = 4
        for node in G.nodes():
            ngh = {}
            if len(G.edges(node)) < N_NEIGHBOURS:
                position = G.nodes[node]['pos']

                for neighbor in G.nodes():
                    if neighbor != node:
                        neighbor_position = G.nodes[neighbor]['pos']
                        dst = distance.euclidean(position, neighbor_position)
                        ngh[dst] = neighbor
                ngh = sorted(ngh.items())
                ngh = ngh[:N_NEIGHBOURS]                          
                for n in ngh:
                    G.add_edge(node, n[1], weight=n[0])


        pos=nx.get_node_attributes(G,'pos')
        nx.draw(G,pos, with_labels=True, font_weight='bold')
        plt.show()

        return G

    def points_from_image(self):
        img = cv2.imread(self.image_path)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        w = cv2.inRange(hsv, (137, 29, 0), (179, 255, 255))

        contours, hierarchy = cv2.findContours(w, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        shape = img.shape
        points = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 10 and h > 10:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                center = (int(x + w / 2), int(shape[1]-(y + h / 2)))
                points.append(center)

        # cv2.imshow("Image", img)
        # cv2.imshow("w", w)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return points


if __name__ == '__main__':
    g = GraphFromImage('map.png')
    g.make_graph()