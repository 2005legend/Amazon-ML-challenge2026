import pytest

H = ["entity_id", "business_name", "business_address", "country"]


def _write(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\t".join(header) + "\n" + "".join("\t".join(r) + "\n" for r in rows), encoding="utf-8")


@pytest.fixture
def tiny_data(tmp_path):
    root = tmp_path / "dataset"
    s1 = [("S1-1", "Ram Marketing Private Limited", "12 MG Road, Pune, Maharashtra", "India"),
          ("S1-2", "Shyam Marketing Private Limited", "4 Station Road, Mumbai, Maharashtra", "India"),
          ("S1-3", "Acme Corp", "1 Main Street, Springfield, IL", "US"),
          ("S1-4", "Gulf Point Corp", "1923 41, Twin Bridges, MT", "US")]
    s2 = [("S2-1", "राम मार्केटिंग प्राइवेट लिमिटेड", "12 MG ROAD, PUNE, महाराष्ट्र", "India"),
          ("S2-2", "श्याम मार्केटिंग प्राइवेट लिमिटेड", "4 STATION ROAD, MUMBAI, महाराष्ट्र", "India"),
          ("S2-3", "ACME CORP", "1 MAIN ST, SPRINGFIELD, IL", "US"),
          ("S2-4", "Acme Corp", "77 OAK AVE, DAYTON, OH", "US")]
    s3 = [("S3-1", "Acme Corporation", "", "US"),
          ("S3-2", "Shyam Marketing Pvt Ltd", "<NULL>, Mumbai, MH", "India")]
    gt = [("S1-1", "S2-1"), ("S1-2", "S2-2,S3-2"), ("S1-3", "S2-3,S3-1"), ("S1-4", "")]
    for split in ("train", "test"):
        extra1 = [("S1-5", "Team Ecole SARL", "175 Boulevard du President Roosevelt, Bordeaux, Nouvelle-Aquitaine",
                   "France")] if split == "test" else []
        extra2 = [("S2-5", "TEAM ECOLE", "175 BD DU PRESIDENT ROOSEVELT, BORDEAUX, Gironde",
                   "France")] if split == "test" else []
        _write(root / split / f"{split}_source1.tsv", H, s1 + extra1)
        _write(root / split / f"{split}_source2.tsv", H, s2 + extra2)
        _write(root / split / f"{split}_source3.tsv", H, s3)
    _write(root / "train" / "train_ground_truth.tsv", ["source1_entity_id", "matched_entity_ids"], gt)
    return root
