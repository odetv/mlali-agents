from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

docs = [
    Document(
        page_content="""
            Kabupaten: Buleleng
            Nama desa wisata: [
            {
                nama desa: Banjar
                kecamatan: Banjar
                kategori: Rintisan
            },
            {
                nama_desa: Banyuseri,
                kecamatan: Banjar,
                kategori: Rintisan
            },
            {
                nama_desa: Cempaga,
                kecamatan: Banjar,
                kategori: Rintisan
            },
            {
                nama_desa: Gobleg,
                kecamatan: Banjar,
                kategori: Rintisan
            },
            {
                nama_desa: Kaliasem,
                kecamatan: Banjar,
                kategori: Rintisan
            },
            {
                nama_desa: Kayuputih,
                kecamatan: Banjar,
                kategori: Maju
            },
            {
                nama_desa: Munduk,
                kecamatan: Banjar,
                kategori: Rintisan
            },
            {
                nama_desa: Pedawa,
                kecamatan: Banjar,
                kategori: Rintisan
            },
            {
                nama_desa: Sidetapa,
                kecamatan: Banjar,
                kategori: Rintisan
            },
            {
                nama_desa: Tigawasa,
                kecamatan: Banjar,
                kategori: Berkembang
            }],
        """,
        metadata={"desc": "DESA WISATA BULELENG"}
    ),
    Document(
        page_content="""
            Kabupaten: Tabanan
            Nama desa wisata: [
                {
                    nama_desa: Antapan,
                    kecamatan: Baturiti,
                    kategori: Berkembang
                },
                {
                    nama_desa: Kukuh,
                    kecamatan: Marga,
                    kategori: Berkembang
                },
                {
                    nama_desa: Tua,
                    kecamatan: Marga,
                    kategori: Berkembang
                },
                {
                    nama_desa: Pinge,
                    kecamatan: Marga,
                    kategori: Berkembang
                },
                {
                    nama_desa: Nyambu,
                    kecamatan: Kediri,
                    kategori: Berkembang
                },
                {
                    nama_desa: Kaba-kaba,
                    kecamatan: Kediri,
                    kategori: Berkembang
                },
                {
                    nama_desa: Mangesta,
                    kecamatan: Penebel,
                    kategori: Berkembang
                },
                {
                    nama_desa: Biaung,
                    kecamatan: Penebel,
                    kategori: Berkembang
                },
                {
                    nama_desa: Jatiluwih,
                    kecamatan: Penebel,
                    kategori: Berkembang
                },
                {
                    nama_desa: Tajen,
                    kecamatan: Penebel,
                    kategori: Berkembang
                },
                {
                    nama_desa: Kerambitan,
                    kecamatan: Kerambitan,
                    kategori: Berkembang
                },
                {
                    nama_desa: Tista,
                    kecamatan: Kerambitan,
                    kategori: Berkembang
                },
                {
                    nama_desa: Antap,
                    kecamatan: Selemadeg,
                    kategori: Berkembang
                },
                {
                    nama_desa: Lalanglinggah,
                    kecamatan: Selemadeg Barat,
                    kategori: Berkembang
                },
                {
                    nama_desa: Gunung Salak,
                    kecamatan: Selemadeg Timur,
                    kategori: Berkembang
                },
                {
                    nama_desa: Belimbing,
                    kecamatan: Pupuan,
                    kategori: Berkembang
                },
                {
                    nama_desa: Bongan,
                    kecamatan: Tabanan,
                    kategori: Berkembang
                },
                {
                    nama_desa: Wanagiri,
                    kecamatan: Selemadeg,
                    kategori: Berkembang
                },
                {
                    nama_desa: Lumbung Kauh,
                    kecamatan: Selemadeg Barat,
                    kategori: Berkembang
                },
                {
                    nama_desa: Megati,
                    kecamatan: Selemadeg Timur,
                    kategori: Berkembang
                },
                {
                    nama_desa: Bantiran,
                    kecamatan: Pupuan,
                    kategori: Berkembang
                },
                {
                    nama_desa: Sanda,
                    kecamatan: Pupuan,
                    kategori: Berkembang
                },
                {
                    nama_desa: Munduk Temu,
                    kecamatan: Pupuan,
                    kategori: Berkembang
                },
                {
                    nama_desa: Cau Belayu,
                    kecamatan: Pupuan,
                    kategori: Berkembang
                },
                {
                    nama_desa: Tegal Mengkeb,
                    kecamatan: Selemadeg Timur,
                    kategori: Berkembang
                },
            ]
        """,
        metadata={"desc": "DESA WISATA TABANAN"}
    ),
    Document(
        page_content="""
            Kabupaten: Badung
            Nama desa wisata: [
                {
                    nama_desa: Bongkasa Pertiwi,
                    kecamatan: Abiansemal,
                    kategori: Berkembang
                },
                {
                    nama_desa: Pangsan,
                    kecamatan: Petang,
                    kategori: Rintisan
                },
                {
                    nama_desa: Petang,
                    kecamatan: Petang,
                    kategori: Berkembang
                },
                {
                    nama_desa: Pelaga,
                    kecamatan: Petang,
                    kategori: Rintisan
                },
                {
                    nama_desa: Belok,
                    kecamatan: Petang,
                    kategori: Rintisan
                },
                {
                    nama_desa: Carangsari,
                    kecamatan: Petang,
                    kategori: Berkembang/maju
                },
                {
                    nama_desa: Sangeh,
                    kecamatan: Abiansemal,
                    kategori: Berkembang
                },
                {
                    nama_desa: Baha,
                    kecamatan: Mengwi,
                    kategori: Berkembang
                },
                {
                    nama_desa: Kapal,
                    kecamatan: Mengwi,
                    kategori: Rintisan
                },
                {
                    nama_desa: Mengwi,
                    kecamatan: Mengwi,
                    kategori: Berkembang
                },
                {
                    nama_desa: Munggu,
                    kecamatan: Mengwi,
                    kategori: Maju
                },
                {
                    nama_desa: Bongkasa,
                    kecamatan: Abiansemal,
                    kategori: Berkembang
                },
                {
                    nama_desa: Abiansemal Dauh Yeh Cani,
                    kecamatan: Abiansemal,
                    kategori: Rintisan
                },
                {
                    nama_desa: Sobangan,
                    kecamatan: Mengwi,
                    kategori: Rintisan
                },
                {
                    nama_desa: Cemagi,
                    kecamatan: Mengwi,
                    kategori: Rintisan
                },
                {
                    nama_desa: Penarungan,
                    kecamatan: Mengwi,
                    kategori: Rintisan
                },
                {
                    nama_desa: Kuwum,
                    kecamatan: Mengwi,
                    kategori: Rintisan
                },
            ]
        """,
        metadata={"desc": "DESA WISATA BADUNG"}
    ),
    Document(
        page_content="""
            Kabupaten: Gianyar
            Nama desa wisata: [
                {
                    nama_desa: Singapadu Tengah,
                    kecamatan: Sukawati,
                    kategori: Maju
                },
                {
                    nama_desa: Singapadu Kaler,
                    kecamatan: Sukawati,
                    kategori: Berkembang
                },
                {
                    nama_desa: Taro,
                    kecamatan: Tegallalang,
                    kategori: Maju
                },
                {
                    nama_desa: Kerta,
                    kecamatan: Payangan,
                    kategori: Berkembang
                },
                {
                    nama_desa: Batubulan,
                    kecamatan: Sukawati,
                    kategori: Maju
                },
                {
                    nama_desa: Kemenuh,
                    kecamatan: Sukawati,
                    kategori: Maju
                },
                {
                    nama_desa: Mas,
                    kecamatan: Ubud,
                    kategori: Mandiri
                },
                {
                    nama_desa: Kendran,
                    kecamatan: Tegallalang,
                    kategori: Berkembang
                },
                {
                    nama_desa: Kedisan,
                    kecamatan: Tegallalang,
                    kategori: Rintisan
                },
                {
                    nama_desa: Keramas,
                    kecamatan: Blahbatuh,
                    kategori: Berkembang
                },
                {
                    nama_desa: Pejeng Kangin,
                    kecamatan: Tampak Siring,
                    kategori: Berkembang
                },
                {
                    nama_desa: Petulu,
                    kecamatan: Ubud,
                    kategori: Berkembang
                },
                {
                    nama_desa: Tegallalang,
                    kecamatan: Tegallalang,
                    kategori: Maju
                },
                {
                    nama_desa: Buahan Kaja,
                    kecamatan: Payangan,
                    kategori: Rintisan
                },
                {
                    nama_desa: Lebih,
                    kecamatan: Gianyar,
                    kategori: Rintisan
                },
                {
                    nama_desa: Sidan,
                    kecamatan: Gianyar,
                    kategori: Rintisan
                },
                {
                    nama_desa: Lodtunduh,
                    kecamatan: Ubud,
                    kategori: Rintisan
                },
                {
                    nama_desa: Singapadu,
                    kecamatan: Sukawati,
                    kategori: Rintisan
                },
                {
                    nama_desa: Celuk,
                    kecamatan: Sukawati,
                    kategori: Rintisan
                },
                {
                    nama_desa: Bedulu,
                    kecamatan: Blahbatuh,
                    kategori: Rintisan
                },
                {
                    nama_desa: Manukaya,
                    kecamatan: Tampak Siring,
                    kategori: Rintisan
                },
                {
                    nama_desa: Sayan,
                    kecamatan: Ubud,
                    kategori: Maju
                },
                {
                    nama_desa: Tampak Siring,
                    kecamatan: Tampak Siring,
                    kategori: Berkembang
                },
                {
                    nama_desa: Kelurahan Beng,
                    kecamatan: Gianyar,
                    kategori: Berkembang
                },
                {
                    nama_desa: Peliatan,
                    kecamatan: Ubud,
                    kategori: Maju
                },
                {
                    nama_desa: Keliki,
                    kecamatan: Tegallalang,
                    kategori: Rintisan
                },
                {
                    nama_desa: Buruan,
                    kecamatan: Blahbatuh,
                    kategori: Rintisan
                },
                {
                    nama_desa: Melinggih Kelod,
                    kecamatan: Payangan,
                    kategori: Maju
                },
                {
                    nama_desa: Pupuan,
                    kecamatan: Blahbatuh,
                    kategori: Rintisan
                },
                {
                    nama_desa: Saba,
                    kecamatan: Blahbatuh,
                    kategori: Rintisan
                },
                {
                    nama_desa: Sebatu,
                    kecamatan: Tegallalang,
                    kategori: Berkembang
                },
                {
                    nama_desa: Batuan,
                    kecamatan: Sukawati,
                    kategori: Berkembang
                },
                {
                    nama_desa: Temesi,
                    kecamatan: Gianyar,
                    kategori: Rintisan
                },
            ]
        """,
        metadata={"desc": "DESA WISATA GIANYAR"}
    ),
    Document(
        page_content="""
            Kabupaten: Klungkung
            Nama desa wisata: [

                {
                    nama_desa: Tihingan,
                    kecamatan: Banjarangkan,
                    kategori: Rintisan
                },

                {
                    nama_desa: Timuhun,
                    kecamatan: Banjarangkan,
                    kategori: Rintisan
                },

                {
                    nama_desa: Bakas,
                    kecamatan: Banjarangkan,
                    kategori: Berkembang
                },

                {
                    nama_desa: Kamasan,
                    kecamatan: Klungkung,
                    kategori: Berkembang
                },

                {
                    nama_desa: Tegak,
                    kecamatan: Klungkung,
                    kategori: Rintisan
                },

                {
                    nama_desa: Gelgel,
                    kecamatan: Klungkung,
                    kategori: Berkembang
                },

                {
                    nama_desa: Besan,
                    kecamatan: Dawan,
                    kategori: Berkembang
                },

                {
                    nama_desa: Pesinggahan,
                    kecamatan: Dawan,
                    kategori: Berkembang
                },

                {
                    nama_desa: Paksebali,
                    kecamatan: Dawan,
                    kategori: Berkembang
                },

                {
                    nama_desa: Jungutbatu,
                    kecamatan: Nusa Penida,
                    kategori: Berkembang
                },

                {
                    nama_desa: Lembongan,
                    kecamatan: Nusa Penida,
                    kategori: Berkembang
                },

                {
                    nama_desa: Ped,
                    kecamatan: Nusa Penida,
                    kategori: Berkembang
                },

                {
                    nama_desa: Batukandik,
                    kecamatan: Nusa Penida,
                    kategori: Berkembang
                },

                {
                    nama_desa: Tanglad,
                    kecamatan: Nusa Penida,
                    kategori: Berkembang
                },

                {
                    nama_desa: Pejukutan,
                    kecamatan: Nusa Penida,
                    kategori: Berkembang
                },

                {
                    nama_desa: Batu Nunggul,
                    kecamatan: Nusa Penida,
                    kategori: Berkembang
                },

                {
                    nama_desa: Kelumpu,
                    kecamatan: Nusa Penida,
                    kategori: Berkembang
                },

                {
                    nama_desa: Suana,
                    kecamatan: Nusa Penida,
                    kategori: Berkembang
                },

                {
                    nama_desa: Aan,
                    kecamatan: Banjarangkan,
                    kategori: Rintisan
                },
            ]
        """,
        metadata={"source": "test"}
    ),
]


# db = FAISS.from_documents(docs, OpenAIEmbeddings())
# db.save_local("src/db/coba_db")

db = FAISS.load_local("src/db/coba_db", OpenAIEmbeddings(), allow_dangerous_deserialization=True)

query = "desa aan"
relevant_response = db.similarity_search_with_relevance_scores(query, k=10)

print(relevant_response)