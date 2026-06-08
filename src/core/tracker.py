import numpy as np


class CentroidTracker:
    def __init__(self, max_disappeared: int = 30):
        self.next_id = 0
        self.objects: dict = {}
        self.disappeared: dict = {}
        self.max_disappeared = max_disappeared

    def update(self, centroids: list) -> dict:
        if not centroids:
            for oid in list(self.disappeared):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    del self.objects[oid]
                    del self.disappeared[oid]
            return {}

        if not self.objects:
            for c in centroids:
                self.objects[self.next_id] = c
                self.disappeared[self.next_id] = 0
                self.next_id += 1
            return dict(zip(range(self.next_id - len(centroids), self.next_id), centroids))

        obj_ids = list(self.objects)
        obj_cents = np.array(list(self.objects.values()), dtype=float)
        new_cents = np.array(centroids, dtype=float)

        D = np.linalg.norm(obj_cents[:, None] - new_cents[None, :], axis=2)
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows, used_cols = set(), set()
        current: dict = {}

        for r, c in zip(rows, cols):
            if r in used_rows or c in used_cols:
                continue
            oid = obj_ids[r]
            self.objects[oid] = centroids[c]
            self.disappeared[oid] = 0
            current[oid] = centroids[c]
            used_rows.add(r)
            used_cols.add(c)

        for r in set(range(len(obj_ids))) - used_rows:
            oid = obj_ids[r]
            self.disappeared[oid] += 1
            if self.disappeared[oid] > self.max_disappeared:
                del self.objects[oid]
                del self.disappeared[oid]

        for c in set(range(len(centroids))) - used_cols:
            self.objects[self.next_id] = centroids[c]
            self.disappeared[self.next_id] = 0
            current[self.next_id] = centroids[c]
            self.next_id += 1

        return current
